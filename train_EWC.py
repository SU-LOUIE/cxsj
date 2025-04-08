import json
import os

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling
from transformers import BitsAndBytesConfig

"""
Mon Mar 31 00:12:56 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.03              Driver Version: 560.35.03      CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA TITAN V                 On  |   00000000:04:00.0 Off |                  N/A |
| 28%   33C    P8             24W /  250W |       1MiB /  12288MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA TITAN V                 On  |   00000000:08:00.0 Off |                  N/A |
| 28%   34C    P8             24W /  250W |       1MiB /  12288MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
W0331 00:13:02.852000 4094061 site-packages/torch/distributed/run.py:792] 
W0331 00:13:02.852000 4094061 site-packages/torch/distributed/run.py:792] *****************************************
W0331 00:13:02.852000 4094061 site-packages/torch/distributed/run.py:792] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W0331 00:13:02.852000 4094061 site-packages/torch/distributed/run.py:792] *****************************************

Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]
Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]
Loading checkpoint shards:  25%|██▌       | 1/4 [00:34<01:43, 34.54s/it]
Loading checkpoint shards:  25%|██▌       | 1/4 [00:34<01:43, 34.54s/it]
Loading checkpoint shards:  50%|█████     | 2/4 [01:08<01:08, 34.26s/it]
Loading checkpoint shards:  50%|█████     | 2/4 [01:08<01:08, 34.26s/it]
Loading checkpoint shards:  75%|███████▌  | 3/4 [01:41<00:33, 33.57s/it]
Loading checkpoint shards:  75%|███████▌  | 3/4 [01:41<00:33, 33.57s/it]
Loading checkpoint shards: 100%|██████████| 4/4 [01:49<00:00, 23.55s/it]
Loading checkpoint shards: 100%|██████████| 4/4 [01:49<00:00, 27.38s/it]

Loading checkpoint shards: 100%|██████████| 4/4 [01:49<00:00, 23.55s/it]
Loading checkpoint shards: 100%|██████████| 4/4 [01:49<00:00, 27.38s/it]
Computing Fisher information matrix...
[rank0]: Traceback (most recent call last):
[rank0]:   File "/home/weixt_lab/cse12212618/train_EWC.py", line 277, in <module>
[rank0]:     main()
[rank0]:   File "/home/weixt_lab/cse12212618/train_EWC.py", line 270, in main
[rank0]:     med_model.compute_fisher(train_data, num_samples=500)
[rank0]:   File "/home/weixt_lab/cse12212618/train_EWC.py", line 128, in compute_fisher
[rank0]:     if p.requires_grad and 'lora' in name.lower()}
[rank0]:                                      ^^^^
[rank0]: UnboundLocalError: cannot access local variable 'name' where it is not associated with a value
[rank0]:[W331 00:16:34.389440787 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
W0331 00:16:36.845000 4094061 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 4096092 closing signal SIGTERM
E0331 00:16:37.112000 4094061 site-packages/torch/distributed/elastic/multiprocessing/api.py:869] failed (exitcode: 1) local_rank: 0 (pid: 4096091) of binary: /home/weixt_lab/cse12212618/.conda/envs/train/bin/python
Traceback (most recent call last):
  File "/home/weixt_lab/cse12212618/.conda/envs/train/bin/torchrun", line 8, in <module>
    sys.exit(main())
             ^^^^^^
  File "/home/weixt_lab/cse12212618/.conda/envs/train/lib/python3.12/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 355, in wrapper
    return f(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/home/weixt_lab/cse12212618/.conda/envs/train/lib/python3.12/site-packages/torch/distributed/run.py", line 918, in main
    run(args)
  File "/home/weixt_lab/cse12212618/.conda/envs/train/lib/python3.12/site-packages/torch/distributed/run.py", line 909, in run
    elastic_launch(
  File "/home/weixt_lab/cse12212618/.conda/envs/train/lib/python3.12/site-packages/torch/distributed/launcher/api.py", line 138, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/weixt_lab/cse12212618/.conda/envs/train/lib/python3.12/site-packages/torch/distributed/launcher/api.py", line 269, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
train_EWC.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-03-31_00:16:36
  host      : titanvgpu01
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 4096091)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================

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
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        # 加载基础模型
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
            
            # 反向传播计算梯度
            loss.backward()

            # 收集梯度并更新Fisher矩阵
            for name, param in trainable_params.items():
                if param.grad is not None:
                    # 跨卡梯度聚合
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    self.fisher[name] += param.grad.pow(2).to(f'cuda:{self.local_rank}')

        # 全局归一化
        for name in self.fisher:
            dist.all_reduce(self.fisher[name], op=dist.ReduceOp.SUM)
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
            gradient_accumulation_steps=16,
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
            save_steps=500
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

if __name__ == "__main__":
    main()