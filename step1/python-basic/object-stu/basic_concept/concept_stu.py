# 不用面向对象
name = "Alice"
age = 18


def introduce(name, age):
    print(f"我是{name}，今年{age}岁")


# 用面向对象
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        print(f"我是{self.name}，今年{self.age}岁")


p = Person("Alice", 18)
p.introduce()

"""
必须掌握这 5 个核心点
class：定义一类东西  
__init__：对象创建时初始化数据  
self：当前这个对象自己  
实例属性：self.name、self.age  
实例方法：对象能做什么，比如 introduce()
"""

"""
Python 不提前声明字段
字段通常在 __init__ 里通过 self.xxx 动态创建
方法第一个参数必须显式写 self
访问属性直接用 obj.name
没有 public/private/protected 关键字
"""
