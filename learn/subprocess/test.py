def step1():
    import subprocess
    def runcmd(command):
        ret = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",
                             timeout=1)
        if ret.returncode == 0:
            print("success:", ret)
        else:
            print("error:", ret)

    runcmd(["dir", "/b"])  # 序列参数
    runcmd("exit 1")  # 字符串参数'

def step2():
    import time
    import subprocess

    def cmd(command):
        subp = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
        subp.wait(2) # wait(timeout): 等待子进程终止。
        if subp.poll() == 0: # 检查进程是否终止，如果终止返回 returncode，否则返回 None。
            print(subp.communicate()[1]) # communicate(input,timeout): 和子进程交互，发送和读取数据。
        else:
            print("失败")

    cmd("ls")
    cmd("exit 1")

if __name__ == '__main__':
    step2()
