# 接收函数作为参数
def multiply(num1, num2):
    return num1 * num2


def method1(func, price, rate):
    return func(price, 1 + rate)


print(method1(multiply, 10, 0.1))

# 返回函数的函数
users = [
    {"name": "john", "age": 25},
    {"name": "jack", "age": 12},
]

# 可以从这里看出 map 和 filter 的区别
# map 是映射，filter 是判断过滤
# map 可以理解为:
# for u in users:
#     names.append(u["name"])
# lambda u: u["name"] 是 lambda 函数
# 等价于
# def get_name(u):
#     return u["name"]
# 所以下面的 map 函数等价于
# map(get_name, users)

# filter 可以理解为:
# for u in users:
#     if u["age"] >= 18:
#         adults.append(u)
names = list(map(lambda u: u["name"], users))
adults = list(filter(lambda u: u["age"] >= 18, users))
sorted_users = sorted(users, key=lambda u: u["age"])
print(names)
print(adults)
print(sorted_users)

# 列表推导式的写法
names = [u["name"] for u in users]
adults = [u for u in users if u["age"] >= 18]
print(names)
print(adults)