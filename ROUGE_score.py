import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from rouge_score import rouge_scorer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import BitsAndBytesConfig

def setup_distributed():
    dist.init_process_group(backend='nccl') 
    local_rank = int(os.environ.get('LOCAL_RANK', 0)) 
    torch.cuda.set_device(local_rank) 
    return local_rank

def cleanup():
    dist.destroy_process_group()  # 结束分布式训练

def main(rank, world_size):
    model_path = './model'
    quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,  
            bnb_8bit_use_double_quant=True,  
            bnb_8bit_compute_dtype=torch.float16  
        )
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=f"cuda:{rank}" if rank >= 0 else "auto",
            quantization_config=quantization_config
        )

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    dataset = load_dataset('./data', 'zh', split='train')
    selected_data = dataset.shuffle(seed=42).select(range(100))

    batch_size_per_gpu = 1  # 调整批量大小以适应显存

    sampler = DistributedSampler(selected_data, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(selected_data, batch_size=batch_size_per_gpu, sampler=sampler)

    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    rouge_scores = []

    for example in dataloader:
        question = example["Question"]
        reference_answer = example["Response"]

        input_text = f"### Question: {question}\n### Response:"
        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {key: value.to(rank) for key, value in inputs.items()}  # 确保输入数据在正确的设备上
        outputs = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_length=512)

        generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        scores = scorer.score(reference_answer, generated_answer)

        rouge_scores.append({
            "question": question,
            "reference_answer": reference_answer,
            "generated_answer": generated_answer,
            "rouge1": {"precision": scores['rouge1'].precision,
                       "recall": scores['rouge1'].recall,
                       "f1": scores['rouge1'].fmeasure},
            "rouge2": {"precision": scores['rouge2'].precision,
                       "recall": scores['rouge2'].recall,
                       "f1": scores['rouge2'].fmeasure},
            "rougeL": {"precision": scores['rougeL'].precision,
                       "recall": scores['rougeL'].recall,
                       "f1": scores['rougeL'].fmeasure},
        })

    output = './rouge_scores.json'
    if rank == 0:
        with open(output, 'w') as f:
            json.dump(rouge_scores, f, indent=4)
        print("Rouge scores saved to", output)

if __name__ == "__main__":
    world_size = torch.cuda.device_count() 
    rank = int(os.environ['RANK'])
    setup_distributed()  
    main(rank, world_size)
    cleanup()  
