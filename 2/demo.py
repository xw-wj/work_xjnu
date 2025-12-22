"""
查看IMDB数据集格式的demo
"""

import pandas as pd
from pathlib import Path

def check_imdb_data():
    """查看IMDB数据集的结构和内容"""
    
    # 数据路径
    data_path = Path("/home/hlj3/wwj/work/2/dataset/imdb/plain_text")
    train_file = data_path / "train-00000-of-00001.parquet"
    test_file = data_path / "test-00000-of-00001.parquet"
    
    print("="*70)
    print("检查IMDB数据集格式")
    print("="*70)
    
    # 读取训练集
    print("\n1. 读取训练集...")
    train_df = pd.read_parquet(train_file)
    
    print(f"\n训练集形状: {train_df.shape}")
    print(f"列名: {train_df.columns.tolist()}")
    print(f"\n数据类型:")
    print(train_df.dtypes)
    
    print(f"\n标签分布:")
    print(train_df['label'].value_counts())
    
    print(f"\n前3条数据:")
    print("-"*70)
    for idx in range(min(3, len(train_df))):
        row = train_df.iloc[idx]
        print(f"\n样本 {idx+1}:")
        print(f"  Label: {row['label']}")
        print(f"  Text (前200字符): {str(row['text'])[:200]}...")
        print(f"  Text 长度: {len(str(row['text']))} 字符")
    
    # 读取测试集
    print("\n" + "="*70)
    print("2. 读取测试集...")
    test_df = pd.read_parquet(test_file)
    
    print(f"\n测试集形状: {test_df.shape}")
    print(f"列名: {test_df.columns.tolist()}")
    
    print(f"\n标签分布:")
    print(test_df['label'].value_counts())
    
    print(f"\n前3条数据:")
    print("-"*70)
    for idx in range(min(3, len(test_df))):
        row = test_df.iloc[idx]
        print(f"\n样本 {idx+1}:")
        print(f"  Label: {row['label']}")
        print(f"  Text (前200字符): {str(row['text'])[:200]}...")
        print(f"  Text 长度: {len(str(row['text']))} 字符")
    
    # 统计信息
    print("\n" + "="*70)
    print("3. 文本长度统计")
    print("="*70)
    
    train_df['text_length'] = train_df['text'].apply(lambda x: len(str(x)))
    test_df['text_length'] = test_df['text'].apply(lambda x: len(str(x)))
    
    print("\n训练集文本长度统计:")
    print(train_df['text_length'].describe())
    
    print("\n测试集文本长度统计:")
    print(test_df['text_length'].describe())
    
    print("\n" + "="*70)
    print("检查完成！")
    print("="*70)
    
    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = check_imdb_data()
    
    print("\n\n关于BERT模型的说明:")
    print("="*70)
    print("bert-base-uncased 是预训练的BERT模型（没有分类头）")
    print("使用 AutoModelForSequenceClassification.from_pretrained() 会:")
    print("  1. 加载 bert-base-uncased 的预训练权重")
    print("  2. 自动添加一个分类头（线性层）用于二分类")
    print("  3. 分类头结构: BertPooler -> Dropout -> Linear(768, 2)")
    print("\n模型结构:")
    print("  BERT Encoder (12层)")
    print("  └── [CLS] token 的输出 (768维)")
    print("      └── 分类头 (768 -> 2)")
    print("="*70)