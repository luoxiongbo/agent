# 列表：多次遍历没问题
my_list = [x for x in range(3)]
print(sum(my_list))  # 3
print(sum(my_list))  # 3（还能再算一遍）

# 生成器：第二次遍历时空的
my_gen = (x for x in range(3))
print(sum(my_gen))  # 3（这里已经耗尽）
print(sum(my_gen))  # 0（因为里面已经没有元素了，不是报错，而是返回0）
print(list(my_gen)) # []（确实是空的）