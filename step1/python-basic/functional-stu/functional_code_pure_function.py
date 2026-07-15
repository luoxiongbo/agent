# 纯函数
# 同样输入永远得到同样输出；
# 不会修改外部状态。
def add_tax(price, rate):
    return price * (1 + rate)

print(add_tax(10, 0.1))