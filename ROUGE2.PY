import json

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = './model'
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

dataset = load_dataset('./data', 'zh', split='train')
selected_data = dataset.shuffle(42).select(range(200))

batch_size = 2
dataloader = DataLoader(selected_data, batch_size=batch_size)

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_scores = []
results = []
for example in dataloader:
    question = example["Question"]
    reference_answer = example["Response"]

    input_text = f"我的问题是: {question}\n### 回答:"
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    outputs = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=512)

    generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print("the answer is:", generated_answer)
    results.append({
        # "question": question,
        "reference_answer": reference_answer,
        "generated_answer": generated_answer,
    })
output = './generate2.json'
with open(output, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4)
print("Rouge scores and average scores saved to", output)