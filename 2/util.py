

import torch
from transformers import PreTrainedModel


def print_trainable_parameters(model: PreTrainedModel) -> None:
    """
    打印模型的可训练参数统计信息
    
    Args:
        model: 预训练模型
    """
    trainable_params = 0
    all_params = 0
    
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    trainable_percent = 100 * trainable_params / all_params
    
    print("="*60)
    print(f"Trainable params: {trainable_params:,} || "
          f"All params: {all_params:,} || "
          f"Trainable%: {trainable_percent:.2f}%")
    print("="*60)


def set_seed(seed: int) -> None:
    """
    设置随机种子以保证实验可重复性
    
    Args:
        seed: 随机种子
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_gpu_info() -> None:
    """打印GPU信息"""
    if torch.cuda.is_available():
        print("="*60)
        print(f"CUDA Available: True")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print("="*60 + "\n")
    else:
        print("="*60)
        print("CUDA Available: False")
        print("Using CPU for training")
        print("="*60 + "\n")