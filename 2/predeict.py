import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 加载模型
model_path = "./outputs/bert_imdb_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# 测试样例
test_reviews = [
    "This movie is absolutely amazing! I loved every minute of it.",
    "Terrible film. Complete waste of time and money.",
    "It was okay, nothing special but entertaining enough."
]

print("="*60)
print("IMDB情感分类预测")
print("="*60)

for review in test_reviews:
    # 编码
    inputs = tokenizer(review, return_tensors="pt", truncation=True, max_length=512)
    
    # 预测
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    
    sentiment = "正面 😊" if pred == 1 else "负面 😔"
    confidence = probs[0][pred].item()
    
    print(f"\n评论: {review}")
    print(f"情感: {sentiment}")
    print(f"置信度: {confidence:.2%}")