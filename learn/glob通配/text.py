import glob
from refile import smart_glob

path = glob.glob(r"/data/pycharm_project/python_learn/Machine_Learning/**/**.py")
print(path)

path = smart_glob("s3://hhb-zhengchaofan-data-processing-oss/tf-labeled-res/20220215_BMK_checked/**/ppl_bag**/**.json")
print(path)
