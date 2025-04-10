import json
import os

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import BitsAndBytesConfig

"""
这一版虽然成功训练完了，但是模型的参数不知道保存到哪里去了
"""
def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return local_rank

class MedicalCoTDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
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

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            for line in data:
                input = line["Question"]
                cot = line["Complex_CoT"]
                outputs = line["Response"]       
                input_seq = self.instruction.format(input)
                target_seq = self.output_ins.format(cot,outputs)
                text = input_seq + target_seq + self.tokenizer.eos_token
                tokenized = self.tokenizer(
                    text,
                    max_length=max_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
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
    
    def __getitem__(self, idx):
        return self.samples[idx]

class MedicalLoRA_EWC_Model:
    def __init__(self, model_path, local_rank):
        self.local_rank = local_rank
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.float16
        # )
        # 选择8bit量化以提高性能，虽然会多占用显存
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,  
            bnb_8bit_use_double_quant=True,  
            bnb_8bit_compute_dtype=torch.float16  
        )
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=f"cuda:{local_rank}" if local_rank >= 0 else "auto",
            quantization_config=quantization_config
        )

        # 配置并应用LoRA
        self.lora_config = LoraConfig(
            r=8,  # 降低秩以减少显存使用
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            bias='lora_only',
            lora_dropout=0.05,
            task_type='CAUSAL_LM'
        )
        self.model = get_peft_model(self.base_model, self.lora_config)
        
        # 显式启用LoRA参数的梯度
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True

        # 分布式训练设置
        if local_rank >= 0:
            self.model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True  # 处理参数未使用问题
            )

        # EWC参数
        self.fisher = {}
        self.optimal_params = {}
        self.ewc_lambda = 1000

    def compute_fisher(self, dataset, num_samples=500):
        # 确保启用梯度计算
        self.model.train()
        
        # 仅收集需要梯度的参数
        trainable_params = {n: p for n, p in self.model.named_parameters() 
                          if p.requires_grad and 'lora' in n.lower()}
        self.fisher = {n: torch.zeros_like(p.data, device=f'cuda:{self.local_rank}') 
                      for n, p in trainable_params.items()}

        # 分布式数据加载
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            pin_memory=True
        )

        samples_per_process = max(1, num_samples // dist.get_world_size())
        
        for i, sample in enumerate(dataloader):
            if i >= samples_per_process:
                break
            
            # 数据移动到当前设备
            inputs = {
                'input_ids': sample['input_ids'].to(f'cuda:{self.local_rank}'),
                'attention_mask': sample['attention_mask'].to(f'cuda:{self.local_rank}'),
                'labels': sample['labels'].to(f'cuda:{self.local_rank}')
            }

            self.model.zero_grad()
            outputs = self.model(**inputs)
            loss = outputs.loss
            loss.backward()

            # 收集梯度并更新Fisher矩阵
            for name, param in trainable_params.items():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    self.fisher[name] += param.grad.pow(2).to(f'cuda:{self.local_rank}')

        # 全局归一化
        for name in self.fisher:
            dist.all_reduce(self.fisher[name], op=dist.ReduceOp.SUM)  # 用这个来聚集所有进程中的梯度
            self.fisher[name] /= num_samples

    def train(self, train_data, eval_data, epochs=3):
        # 保存最优参数
        self.optimal_params = {
            n: p.detach().clone().to(f'cuda:{self.local_rank}')
            for n, p in self.model.named_parameters()
            if p.requires_grad and 'lora' in n.lower()
        }

        # 训练参数配置
        training_args = TrainingArguments(
            output_dir="./medical_lora_ewc",
            per_device_train_batch_size=1,  # 进一步降低batch_size
            gradient_accumulation_steps=8,
            learning_rate=2e-5,
            num_train_epochs=epochs,
            fp16=True,
            local_rank=self.local_rank,
            dataloader_num_workers=2,
            remove_unused_columns=False,
            ddp_find_unused_parameters=True,
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=1000  # 这里是每五百步就保存一次，可以选择性地保存
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

        # 保存模型和tokenizer
        if self.local_rank == 0:
            self.base_model.save_pretrained("./medical_lora_ewc")
            self.tokenizer.save_pretrained("./medical_lora_ewc")
            print("Model and tokenizer saved to ./medical_lora_ewc")

class EWC_Trainer(Trainer):
    def __init__(self, fisher_matrix, optimal_params, ewc_lambda, **kwargs):
        super().__init__(**kwargs)
        self.fisher = fisher_matrix
        self.optimal_params = optimal_params
        self.ewc_lambda = ewc_lambda
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        
        # 计算任务损失
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        task_loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )

        # 计算EWC正则项
        ewc_loss = 0
        if self.fisher:
            for name, param in model.named_parameters():
                if name in self.fisher:
                    ewc_loss += (self.fisher[name] * 
                                (param - self.optimal_params[name]).pow(2)).sum()

        total_loss = task_loss + self.ewc_lambda * ewc_loss
        
        return (total_loss, outputs) if return_outputs else total_loss

def main():
    local_rank = setup_distributed()
    
    model_path = './model'
    data_path = './data/medical_o1_sft_Chinese.json'
    
    med_model = MedicalLoRA_EWC_Model(model_path, local_rank)
    
    dataset = MedicalCoTDataset(data_path, med_model.tokenizer)
    train_size = int(0.9 * len(dataset))
    train_data, eval_data = torch.utils.data.random_split(
        dataset, 
        [train_size, len(dataset)-train_size],
        generator=torch.Generator().manual_seed(42)
    )

    if local_rank == 0:
        print("Computing Fisher information matrix...")
    med_model.compute_fisher(train_data, num_samples=500)
    
    if local_rank == 0:
        print("Starting fine-tuning...")
    med_model.train(train_data, eval_data, epochs=3)

    dist.destroy_process_group()  # 进程结束之后要显示调用这个来释放资源，不然会有警告

if __name__ == "__main__":
    main()