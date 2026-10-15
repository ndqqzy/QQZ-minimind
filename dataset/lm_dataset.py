from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value

#设置tokenizers不并行加速,避免报错
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    # init
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 使用 HuggingFace datasets 的惰性加载，避免一次性读入大文件
        self.samples = load_dataset("json", data_files=data_path, split="train")

    # __len__
    def __len__(self):
        return len(self.samples)
    # __getitem__
    # 我们拿到的是 jsonl 里的每一行
    def __getitem__(self, index):
        sample = self.samples[index]

        # tokenizer 把文本转化为 input_id
        tokens = self.tokenizer(
            str(sample["text"]),  # 假设 jsonl 中的 "text" 字段包含文本内容
            add_special_tokens=False,
            max_length=self.max_length - 2,  # 留出位置给 BOS 和 EOS
            truncation=True,  # 长度超过 max 时自动截断
        ).input_ids
        # 加上 BOS、EOS，并用 PAD 填充
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (
            self.max_length - len(tokens)
        )  # 填充到 max_length
        input_ids = torch.tensor(input_ids, dtype=torch.long)  # 转成 tensor
        # 编写 labels，避免 PAD 参与 loss 计算
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # 非 PAD 位置为 1，PAD 位置为 0
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }





















 