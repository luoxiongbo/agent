import sys

# 列表推导式：一次性生成100万个平方数，占用巨大内存
list_comp = [i * i for i in range(1000000)]
print(sys.getsizeof(list_comp))  # 输出约 8,448,768 字节（约8MB）

# 生成器表达式：只记录算法，几乎不占内存
gen_exp = (i * i for i in range(1000000))
print(sys.getsizeof(gen_exp))    # 输出约 104 字节（几乎可以忽略不计）