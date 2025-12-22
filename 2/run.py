

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint

from data import load_datasets, create_data_collator
from util import print_trainable_parameters, set_seed, print_gpu_info


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """模型相关参数"""
    model_name_or_path: Optional[str] = field(
        default="bert-base-uncased",
        metadata={"help": "预训练模型路径或名称"}
    )


@dataclass
class DataArguments:
    """数据相关参数"""
    data_path: str = field(
        default=None,
        metadata={"help": "IMDB数据集路径"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "最大序列长度"}
    )


def load_model_and_tokenizer(model_args: ModelArguments):
    """
    加载模型和分词器
    
    Args:
        model_args: 模型参数
        
    Returns:
        model, tokenizer
    """
    print("="*60)
    print(f"Loading model: {model_args.model_name_or_path}")
    print("="*60)
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True,
    )
    
    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=2,  # 二分类：正面/负面
        torch_dtype=torch.bfloat16,
    )
    
    # 打印模型参数信息
    print_trainable_parameters(model)
    
    return model, tokenizer


def compute_metrics(eval_pred):
    """
    计算评估指标
    
    Args:
        eval_pred: 包含predictions和label_ids的元组
        
    Returns:
        metrics字典
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def train():
    """主训练函数"""
    
    # 解析命令行参数
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 设置随机种子
    set_seed(training_args.seed)
    
    # 打印GPU信息
    print_gpu_info()
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    
    # 检查checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}")
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(model_args)
    
    # 加载数据集
    train_dataset, eval_dataset = load_datasets(data_args.data_path)
    
    # 创建数据整理器
    data_collator = create_data_collator(tokenizer, data_args.max_length)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # 开始训练
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60 + "\n")
    
    checkpoint = last_checkpoint if last_checkpoint else None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # 保存模型
    trainer.save_model()
    trainer.save_state()
    
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # 最终评估
    if eval_dataset is not None:
        print("\n" + "="*60)
        print("Final Evaluation...")
        print("="*60 + "\n")
        
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
        print(f"\nFinal Accuracy: {metrics['eval_accuracy']:.4f}")
        print(f"Final F1 Score: {metrics['eval_f1']:.4f}")
    
    print("\n" + "="*60)
    print("Training Completed!")
    print(f"Model saved to: {training_args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    train()