import pandas as pd
import pyarrow.parquet as pq

# 文件路径
# 文件路径（改这里）
file_path = "/home/hlj3/wwj/work/4/dataset/data/train-00000-of-00001-ebfa7c1c3a087835.parquet"

print("=" * 80)
print("📦 Parquet 文件信息查看器")
print("=" * 80)

# 1. 使用 pyarrow 查看文件元数据
print("\n【1】文件基本信息：")
print("-" * 80)
parquet_file = pq.read_table(file_path)
print(f"总行数: {parquet_file.num_rows:,}")
print(f"总列数: {parquet_file.num_columns}")
print(f"文件大小: {pd.Series([file_path]).apply(lambda x: f'{pd.io.common.file_exists(x)}')}") 

# 2. 读取数据到 DataFrame
df = pd.read_parquet(file_path)

print("\n【2】列名和数据类型：")
print("-" * 80)
for idx, (col, dtype) in enumerate(df.dtypes.items(), 1):
    print(f"{idx:2d}. {col:30s} - {dtype}")

print("\n【3】数据形状：")
print("-" * 80)
print(f"Shape: {df.shape} (rows × columns)")

print("\n【4】各列的基本统计信息：")
print("-" * 80)
try:
    # 数值列统计
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        print("数值列统计：")
        print(df[numeric_cols].describe())
    else:
        print("没有数值类型的列")
    
    # 文本列的样本信息
    object_cols = df.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        print("\n文本列示例：")
        for col in object_cols:
            try:
                sample = df[col].iloc[0]
                if isinstance(sample, str):
                    print(f"  {col}: {sample[:50]}...")
                else:
                    print(f"  {col}: {type(sample).__name__} 类型")
            except Exception as e:
                print(f"  {col}: 无法显示")
except Exception as e:
    print(f"统计信息生成出错: {e}")

print("\n【5】前5行数据：")
print("=" * 80)
for i in range(min(5, len(df))):
    print(f"\n>>> 第 {i+1} 条数据：")
    print("-" * 80)
    for col in df.columns:
        value = df.iloc[i][col]
        # 如果值太长，截断显示
        if isinstance(value, str) and len(value) > 200:
            display_value = value[:200] + "... (truncated)"
        else:
            display_value = value
        print(f"{col}: {display_value}")

print("\n【6】缺失值统计：")
print("-" * 80)
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("没有缺失值 ✓")

print("\n【7】每列的唯一值数量：")
print("-" * 80)
for col in df.columns:
    try:
        unique_count = df[col].nunique()
        print(f"{col:30s}: {unique_count:,} 个唯一值")
    except TypeError:
        # 处理包含列表/数组的列
        print(f"{col:30s}: (复杂类型，无法计算唯一值)")

print("\n" + "=" * 80)
print("查看完成！✨")
print("=" * 80)

# 额外分析复杂字段
print("\n【8】复杂字段详细分析：")
print("-" * 80)

# 分析 movieGenres
print("\n▶ movieGenres 字段分析：")
try:
    sample_genres = df['movieGenres'].iloc[0]
    print(f"  类型: {type(sample_genres)}")
    print(f"  示例: {sample_genres}")
    print(f"  前5个样本:")
    for i in range(min(5, len(df))):
        print(f"    {i+1}. {df['movieGenres'].iloc[i]}")
except Exception as e:
    print(f"  分析出错: {e}")

# 分析 utterance
print("\n▶ utterance 字段分析：")
try:
    sample_utterance = df['utterance'].iloc[0]
    print(f"  类型: {type(sample_utterance)}")
    if isinstance(sample_utterance, dict):
        print(f"  字典键: {list(sample_utterance.keys())}")
        for key, value in sample_utterance.items():
            print(f"    - {key}: {type(value).__name__}, 长度={len(value) if hasattr(value, '__len__') else 'N/A'}")
            if hasattr(value, '__len__') and len(value) > 0:
                print(f"      首个元素示例: {value[0][:100] if isinstance(value[0], str) else value[0]}...")
except Exception as e:
    print(f"  分析出错: {e}")

print("\n" + "=" * 80)