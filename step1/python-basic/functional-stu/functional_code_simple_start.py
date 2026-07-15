def square(num):
    return num * num

def apply(func, *args):
    return func(*args)

print(square(3))
print(apply(square, 3))