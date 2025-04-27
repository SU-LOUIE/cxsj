import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bert_score import score
import jieba

# 假设模型文件所在路径
model_path = "./model"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, torch_dtype=torch.float32)

def tokenize_zh(text: str):
    return " ".join(jieba.lcut(text))

quantization_config = torch.bfloat16  # 这里示例用 bfloat16 量化，可按需调整
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=quantization_config,
    device_map="auto"
)

input_text = "根据描述，一个1岁的孩子在夏季头皮出现多处小结节，长期不愈合，且现在疮大如梅，溃破流脓，口不收敛，头皮下有空洞，患处皮肤增厚。这种病症在中医中诊断为什么病？"
inputs = tokenizer(input_text, return_tensors="pt")

inputs = {key: value.to(model.device) for key, value in inputs.items()}

outputs = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=512)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
reference_text = '从中医的角度来看，你所描述的症状符合“蝼蛄疖”的病症。这种病症通常发生在头皮，表现为多处结节，溃破流脓，形成空洞，患处皮肤增厚且长期不愈合。湿热较重的夏季更容易导致这种病症的发展，特别是在免疫力较弱的儿童身上。建议结合中医的清热解毒、祛湿消肿的治疗方法进行处理，并配合专业的医疗建议进行详细诊断和治疗。'

# 不需要使用 jieba 分词，因为 bertscore 会处理文本
preds = [generated_text]
refs = [reference_text]

# 计算 BERTScore
P, R, F1 = score(preds, refs, lang="zh", model_type="bert-base-chinese")

# 打印 BERTScore
print(f"Precision: {P.mean().item()}")
print(f"Recall: {R.mean().item()}")
print(f"F1: {F1.mean().item()}")