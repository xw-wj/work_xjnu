import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


@dataclass
class IMDBDataOutput:
    """IMDB数据输出格式"""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class IMDBDataset(Dataset):
    """IMDB情感分类数据集"""
    
    def __init__(self, data_path: str, split: str = "train") -> None:
        """
        初始化数据集
        
        Args:
            data_path: 数据集根目录
            split: 数据集划分，'train' 或 'test'
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.split = split
        
        # 加载数据
        self.data = self.load_data()
        
    def load_data(self) -> pd.DataFrame:
        """加载parquet数据文件"""
        if self.split == "train":
            file_path = self.data_path / "train-00000-of-00001.parquet"
        elif self.split == "test":
            file_path = self.data_path / "test-00000-of-00001.parquet"
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train' or 'test'")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        data = pd.read_parquet(file_path)
        print(f"Loaded {self.split} dataset: {len(data)} samples")
        print(f"Label distribution: {data['label'].value_counts().to_dict()}")
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[str, int]:
        """
        获取单个样本
        
        Returns:
            text: 评论文本
            label: 情感标签 (0: 负面, 1: 正面)
        """
        row = self.data.iloc[index]
        text = str(row['text'])
        label = int(row['label'])
        
        return text, label


class IMDBDataCollator:
    """
    IMDB数据整理器
    将多个样本整理成batch
    """
    
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 512) -> None:
        """
        初始化数据整理器
        
        Args:
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, features: List[Tuple[str, int]]) -> Dict[str, torch.Tensor]:
        """
        将一个batch的样本整理成模型输入格式
        
        Args:
            features: [(text, label), ...] 格式的样本列表
            
        Returns:
            包含input_ids, attention_mask, labels的字典
        """
        texts = [f[0] for f in features]
        labels = [f[1] for f in features]
        
        # 批量编码文本
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long)
        }


def load_datasets(data_path: str) -> Tuple[IMDBDataset, IMDBDataset]:
    """
    加载训练集和测试集
    
    Args:
        data_path: 数据集根目录
        
    Returns:
        train_dataset, test_dataset
    """
    print("="*60)
    print("Loading IMDB Dataset...")
    print("="*60)
    
    train_dataset = IMDBDataset(data_path, split="train")
    test_dataset = IMDBDataset(data_path, split="test")
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print("="*60 + "\n")
    
    return train_dataset, test_dataset


def create_data_collator(tokenizer: AutoTokenizer, max_length: int = 512) -> IMDBDataCollator:
    """
    创建数据整理器
    
    Args:
        tokenizer: 分词器
        max_length: 最大序列长度
        
    Returns:
        IMDBDataCollator实例
    """
    return IMDBDataCollator(tokenizer, max_length)