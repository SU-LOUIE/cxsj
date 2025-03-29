from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoConfig
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model
import json
import random


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 现在是这个json读取的格式有一点问题，还需要修改
"""
The `load_in_4bit` and `load_in_8bit` arguments are deprecated and will be removed in the future versions. Please, pass a `BitsAndBytesConfig` object in `quantization_config` argument instead.

Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]
Loading checkpoint shards:  25%|██▌       | 1/4 [00:05<00:16,  5.42s/it]
Loading checkpoint shards:  50%|█████     | 2/4 [00:10<00:10,  5.48s/it]
Loading checkpoint shards:  75%|███████▌  | 3/4 [00:16<00:05,  5.58s/it]
Loading checkpoint shards: 100%|██████████| 4/4 [00:18<00:00,  3.93s/it]
Loading checkpoint shards: 100%|██████████| 4/4 [00:18<00:00,  4.51s/it]
Traceback (most recent call last):
  File "/home/weixt_lab/cse12212618/train_EWC.py", line 220, in <module>
    dataset = MedicalCoTDataset(data_path, med_model.tokenizer)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/weixt_lab/cse12212618/train_EWC.py", line 42, in __init__
    data = json.loads(line)
           ^^^^^^^^^^^^^^^^
  File "/home/weixt_lab/cse12212618/.conda/envs/train/lib/python3.12/json/__init__.py", line 346, in loads
    return _default_decoder.decode(s)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/weixt_lab/cse12212618/.conda/envs/train/lib/python3.12/json/decoder.py", line 337, in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/weixt_lab/cse12212618/.conda/envs/train/lib/python3.12/json/decoder.py", line 355, in raw_decode
    raise JSONDecodeError("Expecting value", s, err.value) from None
json.decoder.JSONDecodeError: Expecting value: line 2 column 1 (char 2)
"""

class MedicalCoTDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length = 512):
        super().__init__()
        self.tokenizer = tokenizer
        self.samples = []
        self.instruction = """Below is an instruction that describes a task, paired with an input that provides further context.
        Write a response that appropriately completes the request.
        Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

        ### Instruction:
        You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
        Please answer the following medical question.

        ### Question:
        {}
        """
        self.output_ins = """### Response:
        <think>
        {}
        </think>
        {}"""

        """### Important Notice

        It's crucial to add the EOS (End of Sequence) token at the end of each training dataset entry, otherwise you may encounter infinite generations.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                input = data["Question"]
                cot = data["Complex_CoT"]
                outputs = data["Response"]       
                input_seq = self.instruction.format(input)
                target_seq = self.output_ins.format(cot,outputs)
                text = input_seq + target_seq + self.tokenizer.eos_token
                tokenized = self.tokenizer(
                    text,
                    max_length = max_length,
                    truncation = True,
                    padding = 'max_length',
                    return_tensors = 'pt'
                )
                input_len = len(self.tokenizer(input_seq)['input_ids'])
                labels = tokenized['input_ids'].clone()[0]
                labels[:input_len] = -100
                self.samples.append({
                    'input_ids': tokenized['input_ids'][0],
                    'attention_mask': tokenized['attention_mask'][0],
                    'labels': labels
                })
    def __len__(self):
        return len(self.samples)
    def __getitems__(self, keys):
        return self.samples[keys]

class MedicalLoRA_EWC_Model:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit = True
        )
        self.lora_config = LoraConfig(
            r = 32, # 在这里增大秩可能可以缓解灾难性遗忘的问题
            lora_alpha=32,
            target_modules=["q_proj","v_proj"],  # 在论文里面可以看到这两个对结果的影响最大
            bias='lora_only',
            lora_dropout=0.05,
            task_type='CAUSAL_LM'
        )
        self.model = get_peft_model(self.model,self.lora_config)

        # 下面是EWC的一些参数
        self.fisher = {}
        self.optimal_params = {}
        self.ewc_lambda = 1000
        self.model.to(device)
    
    def compute_fisher(self, dataset, num_samples=500):
        # 计算fisher信息矩阵
        self.model.eval()
        self.fisher = {n: torch.zeros_like(p).to(device) 
                      for n, p in self.model.named_parameters() 
                      if p.requires_grad and 'lora' in n}
        
        for _ in range(num_samples):
            sample = random.choice(dataset)
            inputs = {
                'input_ids': sample['input_ids'].unsqueeze(0).to(device),
                'attention_mask': sample['attention_mask'].unsqueeze(0).to(device),
                'labels': sample['labels'].unsqueeze(0).to(device)
            }
            
            self.model.zero_grad()
            outputs = self.model(**inputs)
            loss = outputs.loss
            loss.backward()
            
            # 更新Fisher信息
            for name, param in self.model.named_parameters():
                if name in self.fisher and param.grad is not None:
                    self.fisher[name] += param.grad.pow(2)
        
        # 归一化
        for name in self.fisher:
            self.fisher[name] /= num_samples    

    def train(self, train_data, eval_data, epochs=3):
        # 单任务训练（自动应用EWC）, 如果是多训练的话要根据上一次的训练去算fisher矩阵
        self.optimal_params = {
            n: p.detach().clone().to(device)
            for n, p in self.model.named_parameters()
            if p.requires_grad and 'lora' in n
        }
        
        # 训练参数配置
        training_args = TrainingArguments(
            output_dir="./medical_lora_ewc",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            num_train_epochs=epochs,
            fp16=True,
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=200,
            save_strategy="no"
        )
        
        trainer = EWC_Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            ),
            fisher_matrix=self.fisher,
            optimal_params=self.optimal_params,
            ewc_lambda=self.ewc_lambda
        )
        
        trainer.train()

class EWC_Trainer(Trainer):
    def __init__(self, fisher_matrix, optimal_params, ewc_lambda, **kwargs):
        super().__init__(**kwargs)
        self.fisher = fisher_matrix
        self.optimal_params = optimal_params
        self.ewc_lambda = ewc_lambda
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # 基础任务损失
        outputs = model(**inputs)
        task_loss = outputs.loss
        
        # EWC正则化项
        ewc_loss = 0
        if self.fisher:
            for name, param in model.named_parameters():
                if name in self.fisher:
                    ewc_loss += (self.fisher[name] * 
                                (param - self.optimal_params[name]).pow(2)).sum()
        
        total_loss = task_loss + self.ewc_lambda * ewc_loss
        
        return (total_loss, outputs) if return_outputs else total_loss

def generate_response(model, question, max_length=300):
    input_seq = f"{model.instruction}\n问题：{question}\n思考："
    inputs = model.tokenizer(input_seq, return_tensors="pt").to(device)
    
    outputs = model.model.generate(
        inputs.input_ids,
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=model.tokenizer.eos_token_id
    )
    
    full_response = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_part = full_response[len(input_seq):]
    
    if "答案：" in generated_part:
        reasoning, answer = generated_part.split("答案：", 1)
        return reasoning.strip(), answer.strip()
    else:
        return generated_part, ""  
          
if __name__ == "__main__":
    model_path = './model'
    data_path = './data/medical_o1_sft_Chinese.json'
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # tokenizer.pad_token = tokenizer.eos_token


    # EOS_TOKEN = tokenizer.eos_token    
    med_model = MedicalLoRA_EWC_Model(model_path)
    
    dataset = MedicalCoTDataset(data_path, med_model.tokenizer)
    train_size = int(0.9 * len(dataset))
    train_data, eval_data = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    
    print("计算Fisher信息矩阵...")
    med_model.compute_fisher(train_data, num_samples=500)
    
    print("开始微调训练...")
    med_model.train(train_data, eval_data, epochs=3)

    # test_question = "糖尿病患者应该如何安排饮食？"
    # reasoning, answer = generate_response(med_model, test_question)
    
    # print("\n=== 测试问题 ===")
    # print(test_question)
    # print("\n=== 推理过程 ===")
    # print(reasoning)
    # print("\n=== 最终答案 ===")
    # print(answer)    