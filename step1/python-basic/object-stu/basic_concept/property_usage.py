class Person:
    def __init__(self, age):
        self.age = age

    @property
    def age(self):
        print("get age value")
        return self._age

    @age.setter
    def age(self, value):
        print("set age value")
        if value < 0:
            raise ValueError("age cannot be negative")
        self._age = value

p = Person(18)
p.age = 20
print(p.age)
