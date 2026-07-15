import pandas as pd

data = {
    "name": ["Alice", "Bob", "Charlie", "David"],
    "age": [25, 30, 35, 28],
    "city": ["Beijing", "Shanghai", "Shenzhen", "Hangzhou"],
    "salary": [10000, 15000, 20000, 12000]
}

df = pd.DataFrame(data)

print(df)
print(50*'-')
print(df.head())
print(50*'-')
print(df.shape)
print(50*'-')
print(df.columns)
print(50*'-')
print(df.info())
print(50*'-')
print(df.describe())
print(50*'-')

print(df["name"])
print(50*'-')
print(df[["name", "salary"]])
print(50*'-')
# 第 0 行
print(df.iloc[0])
print(50*'-')
# 位置
print(df.iloc[0, 1])
print(50*'-')
