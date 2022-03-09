import time


def cal_time(func):
    def wrapper(*args, **kwargs):
        begin = time.time()  # 给旧功能加新功能
        func(*args, **kwargs)
        end = time.time()
        print(end - begin)

    return wrapper


@cal_time
def solution():
    print('程序开始执行')
    time.sleep(2)


# 带参数的装饰器
def caltime(user='zhengchaofan'):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(user)
            begin = time.time()  # 给旧功能加新功能
            func(*args, **kwargs)
            end = time.time()
            print(end - begin)

        return wrapper

    return decorator


@caltime(user='Aiden')
def solution2():
    print('程序开始执行')
    time.sleep(2)


if __name__ == '__main__':
    solution()
