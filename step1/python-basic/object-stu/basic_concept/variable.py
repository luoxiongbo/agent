# 其实就跟 Java 里的 String 一样，不可变对象
x = 1
y = x
y = 2

print(x)
print(y)

a = [1, 2]
b = a
b.append(3)

print(a)

a = [1, 2, 3]
b = a

b = [4, 5, 6]

print(a)  # [1, 2, 3]
print(b)  # [4, 5, 6]

a = [1, 2, 3]
b = a

b.append(4)

print(a)  # [1, 2, 3, 4]
print(b)  # [1, 2, 3, 4]
