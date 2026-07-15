from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
"""
传统 NLP 例子：判断一句话是正面还是负面情绪。
它的流程是：
把文本变成 词/字符 特征
把特征转成 数字向量
用 分类模型 学习“正面/负面”的规律
对新句子做预测
"""

# 训练数据：句子 + 标签
texts = [
    "这个产品很好用",
    "体验非常棒",
    "我很喜欢这个功能",
    "客服响应很快",
    "这个东西太难用了",
    "体验很差",
    "我不喜欢这个产品",
    "客服一直不回复",
]

labels = [
    "positive",
    "positive",
    "positive",
    "positive",
    "negative",
    "negative",
    "negative",
    "negative",
]

"""
这里的 CountVectorizer 做的是传统 NLP 里的“特征工程”：把文本拆成字符或词，然后统计出现次数。
e.g.
这个功能很好用
->
这、个、功、能、很、好、用、这个、功能、很好、好用
->
然后变成类似这样的数字向量：
[1, 1, 0, 1, 2, 0, ...]

分类器并不真正“理解”这句话，它只是学到：
“好用”“喜欢”“棒”“快” 更常出现在 positive 里
“差”“难用”“不喜欢”“不回复” 更常出现在 negative 里
这就是传统 NLP 的典型思路：文本特征化 + 机器学习模型。
"""
# 构建传统 NLP Pipeline
model = Pipeline([
    # 中文这里用字符 n-gram，避免依赖分词工具
    ("vectorizer", CountVectorizer(analyzer="char", ngram_range=(1, 2))),
    ("classifier", MultinomialNB())
])

# 训练模型
model.fit(texts, labels)

# 测试新句子
test_texts = [
    "这个功能很好用",
    "这个产品太差了",
    "客服回复很快",
    "体验不好",
]

for text in test_texts:
    prediction = model.predict([text])[0]
    print(text, "=>", prediction)