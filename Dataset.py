from torch.utils.data import Dataset
import json

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

from transformers import AutoTokenizer
data_path = 'D:/pycharm/code/data/medical_o1_sft_Chinese.json'
model_path = 'D:/model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pos_token = tokenizer.eos_token
dataset = MedicalCoTDataset(data_path,tokenizer)