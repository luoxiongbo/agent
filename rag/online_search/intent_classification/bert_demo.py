import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


# 1. 加载中文 BERT
model_name = "google-bert/bert-base-chinese"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# 切换到推理模式
model.eval()


# 2. 准备带有 [MASK] 的句子
text = "我喜欢吃[MASK]。"

inputs = tokenizer(text, return_tensors="pt")

print("原始文本：", text)
print("分词结果：", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
print("input_ids：", inputs["input_ids"])


# 3. 找出 [MASK] 所在的位置
mask_positions = (
    inputs["input_ids"] == tokenizer.mask_token_id
).nonzero(as_tuple=True)

mask_position = mask_positions[1].item()

print("[MASK] 的位置：", mask_position)


# 4. 执行 BERT 前向传播
with torch.no_grad():
    outputs = model(**inputs)

# logits.shape:
# [batch_size, sequence_length, vocabulary_size]
logits = outputs.logits

print("logits.shape：", logits.shape)


# 5. 取出 [MASK] 位置对整个词表的预测分数
mask_logits = logits[0, mask_position]

# 取分数最高的 5 个 token
top5_token_ids = torch.topk(mask_logits, k=5).indices.tolist()


# 6. 输出预测结果
print("\nBERT 预测结果：")

for token_id in top5_token_ids:
    predicted_token = tokenizer.decode([token_id])

    completed_text = text.replace(
        tokenizer.mask_token,
        predicted_token,
    )

    score = mask_logits[token_id].item()

    print(
        f"预测 token：{predicted_token}，"
        f"得分：{score:.3f}，"
        f"完整句子：{completed_text}"
    )