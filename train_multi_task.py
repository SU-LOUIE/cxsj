"""
    因为遇到了灾难性遗忘的问题，现在有两种解决的方案
    一是使用multi-task，通过任务特定的适配层来
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from collections import deque
import numpy as np
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'D:/model'
data_path = 'D:/pycharm/code/data'
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype = torch.float16,
    device_map = 'auto',
)

class MultiTaskLlama(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.lora_config = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=["q_proj","v_proj"],
            lora_dropout=0.1,
            task_type='CAUSAL_LM',
        )
        self.medical_proj = nn.Linear(4096,4096)
        self.dialogue_proj = nn.Linear(4096,4096)
        self.llama = get_peft_model(self.llama,self.lora_config)

    def forward(self, input_ids, attention_mask, task_type):
        outputs = self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = outputs.hidden_states[-1]

        if task_type == "medical":
            adapted = self.medical_proj(hidden_states)
        elif task_type == "dialogue":
            adapted = self.dialogue_proj(hidden_states)
        else:
            raise ValueError("Unknown task type")

        logits = self.llama.lm_head(adapted)
        return logits

model = MultiTaskLlama(model).to(device)
# 按照医疗数据集的格式加上提示词，但是通用类型的不一定能用
train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.

### Question:
{}

### Response:
<think>
{}
</think>
{}"""

"""### Important Notice

It's crucial to add the EOS (End of Sequence) token at the end of each training dataset entry, otherwise you may encounter infinite generations.
"""

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


def formatting_prompts_func(examples):
    inputs = examples["Question"]
    cots = examples["Complex_CoT"]
    outputs = examples["Response"]
    texts = []
    for input, cot, output in zip(inputs, cots, outputs):
        text = train_prompt_style.format(input, cot, output) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }
class MultiTaskDataset(Dataset):
    def __init__(self, medical_data, dialogue_data, tokenizer, max_length=2048):
        self.medical_data = medical_data
        self.dialogue_data = dialogue_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return max(len(self.medical_data), len(self.dialogue_data))
    
    def __getitem__(self, idx):
        # 交替采样
        if idx % 2 == 0:
            data = self.medical_data[idx // 2 % len(self.medical_data)]
            task_type = "medical"
        else:
            data = self.dialogue_data[idx // 2 % len(self.dialogue_data)]
            task_type = "dialogue"

        formatted_text = formatting_prompts_func(data["text"], data["label"], task_type)
        
        encoding = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        encoding["labels"] = encoding["input_ids"].clone()
        encoding["task_type"] = task_type
        return encoding
    
medical_data = [{"text": "patient with fever and cough", "label": "flu"}, ...]
dialogue_data = [{"text": "Hello, how are you?", "label": "I'm good, thanks!"}, ...]

dataset = MultiTaskDataset(medical_data, dialogue_data, tokenizer)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
replay_buffer = deque(maxlen=1000)
for data in dialogue_data[:100]:  # 初始填充
    replay_buffer.append(data)

def train_epoch(model, dataloader, optimizer, epoch, total_epochs=3):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        # 动态任务权重
        medical_prob = 0.5 + 0.3 * (epoch / total_epochs)
        task_type = "medical" if np.random.rand() < medical_prob else "dialogue"
        
        # 从回放缓冲中采样
        if task_type == "medical" and len(replay_buffer) > 0:
            replay_samples = random.sample(replay_buffer, 1)
            replay_batch = dataset.collate_fn([dataset.format_sample(s, "dialogue") for s in replay_samples])
            batch = {k: torch.cat([v, replay_batch[k]]) for k, v in batch.items()}
        
        # 准备输入
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "labels": batch["labels"].to(device)
        }
        
        # 前向传播
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            task_type=task_type
        )

        shift_logits = outputs[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
        # 更新回放缓冲
        if task_type == "dialogue":
            for i in range(len(batch["input_ids"])):
                replay_buffer.append({
                    "text": tokenizer.decode(batch["input_ids"][i]), 
                    "label": tokenizer.decode(batch["labels"][i])
                })
    
    return total_loss / len(dataloader)

for epoch in range(3):
    avg_loss = train_epoch(model, train_loader, optimizer, epoch)
    print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

model.save_pretrained("./multitask_llama")
tokenizer.save_pretrained("./multitask_llama")