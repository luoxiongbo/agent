class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        print("动物在叫")

class Dog(Animal):
    def speak(self):
        print(f"{self.name}：汪汪")

dog = Dog("小黑")
dog.speak()