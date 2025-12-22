import pandas as pd
import json
from pathlib import Path

# 读取原始数据
input_file = "dataset/data/train-00000-of-00001-ebfa7c1c3a087835.parquet"
df = pd.read_parquet(input_file)

print(f"原始数据集大小: {len(df)} 条记录")
print("=" * 80)

# ============================================================================
# 方案1: 转换为 ShareGPT 格式（推荐，支持多轮对话）
# ============================================================================
def convert_to_sharegpt():
    """
    转换为 ShareGPT 格式的多轮对话数据
    每个 utterance 包含多轮对话，按顺序交替分配给 human 和 gpt
    """
    sharegpt_data = []
    
    for idx, row in df.iterrows():
        utterance = row['utterance']
        
        # 跳过无效数据
        if not isinstance(utterance, dict) or 'lines' not in utterance:
            continue
        
        lines = utterance['lines']
        
        # 至少需要2条对话（一问一答）
        if len(lines) < 2:
            continue
        
        # 构建对话列表
        conversations = []
        for i, line in enumerate(lines):
            # 清理文本：去除人物名称标签（如 "BIANCA\n"）
            text = line.strip()
            
            # 方法1: 去除开头的大写人名（匹配模式：全大写单词 + 换行）
            import re
            text = re.sub(r'^[A-Z][A-Z\s]+\n', '', text)
            
            # 方法2: 如果人名后面有换行，去除第一行
            # if '\n' in text:
            #     lines_split = text.split('\n', 1)
            #     if lines_split[0].isupper():  # 第一行全大写
            #         text = lines_split[1] if len(lines_split) > 1 else text
            
            # 再次清理多余空白
            text = text.strip()
            
            # 跳过空文本
            if not text:
                continue
            
            # 奇数位置是 human，偶数位置是 gpt
            role = "human" if i % 2 == 0 else "gpt"
            
            conversations.append({
                "from": role,
                "value": text
            })
        
        # ============================================================
        # 系统提示词选项（根据需求选择）
        # ============================================================
        
        # 选项1: 不使用系统提示词（推荐 - 训练通用对话模型）
        sharegpt_data.append({
            "conversations": conversations
        })
        
        # 选项2: 使用电影信息作为背景（训练电影对话风格模型）
        # system_prompt = f"这是一段来自电影《{row['movieTitle'].strip()}》({row['movieYear'].strip()})的对话。"
        # sharegpt_data.append({
        #     "conversations": conversations,
        #     "system": system_prompt
        # })
        
        # 选项3: 使用通用对话系统提示词
        # sharegpt_data.append({
        #     "conversations": conversations,
        #     "system": "你是一个友好、自然的对话助手。"
        # })
    
    return sharegpt_data

# ============================================================================
# 方案2: 转换为 Alpaca 格式（单轮对话）
# ============================================================================
def convert_to_alpaca():
    """
    转换为 Alpaca 格式
    将每个 utterance 的第一句作为 instruction，最后一句作为 output
    """
    alpaca_data = []
    
    for idx, row in df.iterrows():
        utterance = row['utterance']
        
        if not isinstance(utterance, dict) or 'lines' not in utterance:
            continue
        
        lines = utterance['lines']
        
        if len(lines) < 2:
            continue
        
        # 第一句作为人类指令
        instruction = lines[0].strip()
        
        # 最后一句作为模型回答
        output = lines[-1].strip()
        
        # 中间的对话作为历史记录
        history = []
        for i in range(1, len(lines) - 1, 2):
            if i + 1 < len(lines):
                history.append([
                    lines[i].strip(),
                    lines[i + 1].strip()
                ])
        
        alpaca_data.append({
            "instruction": instruction,
            "input": "",
            "output": output,
            "system": f"这是一段来自电影《{row['movieTitle'].strip()}》的对话。",
            "history": history if history else []
        })
    
    return alpaca_data

# ============================================================================
# 执行转换
# ============================================================================
print("\n选择转换格式：")
print("1. ShareGPT 格式（推荐，完整保留多轮对话）")
print("2. Alpaca 格式（提取首尾对话，中间作为历史）")
print()

# 默认使用 ShareGPT 格式
use_sharegpt = True

if use_sharegpt:
    print("✓ 使用 ShareGPT 格式转换...")
    converted_data = convert_to_sharegpt()
    output_file = "dataset/data/train_sharegpt.json"
    format_name = "sharegpt"
else:
    print("✓ 使用 Alpaca 格式转换...")
    converted_data = convert_to_alpaca()
    output_file = "dataset/data/train_alpaca.json"
    format_name = "alpaca"

# 保存转换后的数据
Path(output_file).parent.mkdir(parents=True, exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=2)

print(f"\n✅ 转换完成！")
print(f"   输出文件: {output_file}")
print(f"   转换后数据量: {len(converted_data)} 条")
print(f"   原始数据量: {len(df)} 条")
print()

# ============================================================================
# 生成 dataset_info.json 配置
# ============================================================================
dataset_info = {}

if use_sharegpt:
    dataset_info["movie_dialogue_sharegpt"] = {
        "file_name": "train_sharegpt.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations"
            # 如果使用了 system 字段，取消下面这行的注释：
            # "system": "system"
        }
    }
else:
    dataset_info["movie_dialogue_alpaca"] = {
        "file_name": "train_alpaca.json",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "system": "system",
            "history": "history"
        }
    }

# 保存 dataset_info.json
dataset_info_file = "dataset/dataset_info.json"
with open(dataset_info_file, 'w', encoding='utf-8') as f:
    json.dump(dataset_info, f, ensure_ascii=False, indent=2)

print(f"📋 已生成 dataset_info.json 配置文件: {dataset_info_file}")
print()
print("=" * 80)
print("📖 使用说明：")
print("=" * 80)
print(f"1. 将生成的文件复制到 LLaMA Factory 的 data 目录")
print(f"2. 将 dataset_info.json 的内容添加到 LLaMA Factory 的 data/dataset_info.json")
print(f"3. 在训练时使用数据集名称: movie_dialogue_{format_name}")
print()

# 显示前3条转换后的数据样例
print("=" * 80)
print("📝 转换后的数据样例（前3条）：")
print("=" * 80)
for i, item in enumerate(converted_data[:3], 1):
    print(f"\n第 {i} 条数据：")
    print(json.dumps(item, ensure_ascii=False, indent=2))
    print("-" * 80)