from pprint import pprint
from random import randint
import pandas as pd
import matplotlib.pyplot as plt


def run():
    df = pd.DataFrame([['姓名', '班级', '年龄'],['zhencghaofan', '1601', '22'],['zhengkeke', '1602', '18']])
    df.to_csv('all.csv')


if __name__ == '__main__':
    run()
