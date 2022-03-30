"""
__repr__
    对对象进行重写，默认：类名 + obejct at + 内存地址
"""


class Person():
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __call__(self, *args, **kwargs):
        dict01 = dict(**kwargs)
        print(f"Person {self.name} doing job {dict01['job']}")

    def __repr__(self):
        return 'Person类，包含name=' + self.name + '和age=' + str(self.age) + '两个实例属性'


if __name__ == '__main__':
    p = Person(name='Zhengchaofan',age=99)
    p(job="taichi")  # Person Zhengchaofan doing job taichi
    print(p)  # Person类，包含name=Zhengchaofan和age=99两个实例属性
    res = p

