from loguru import logger
import time

logger.add(f'auto_log/runtime_{time.time()}.log')


@logger.catch
def test(x, y, z):
    print('begin')  # print 不能被获取到
    return 1 / (x + y + z)


def test2():
    # 都会记录到
    logger.debug("debug message")
    logger.info("info level message")
    logger.warning("warning level message")
    logger.critical("critical level message")


if __name__ == '__main__':
    # test(1, 0, 0)
    test2()
