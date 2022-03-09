import csv
from refile import smart_open
import pdb
from pprint import pprint
import pandas as pd

def csv_read_csv():
    # csv模块读写CSV文件
    inpath = f's3://tf-shared-data/parse_data/ppl_bag_20211120_143125/timestamp_align_merged.csv'
    out_path = "s3://hhb-zhengchaofan-data-processing-oss/test/test.csv"
    with smart_open(inpath, "r") as incsv, smart_open(out_path, "w",) as outcsv:
        freader = csv.reader(incsv, delimiter=",")
        # for row in freader:  # 迭代行
        #     print(row)   # ['frame_id', 'left_lidar', 'middle_lidar', 'right_lidar', 'fuser_lidar', '#cam_10#image_raw#compressed', '#cam_11#image_raw#compressed', '#cam_15#image_raw#compressed', '#cam_2#image_raw#compressed', '#cam_5#image_raw#compressed', 'corr_imu/#sensor#corr_imu', 'ins/#sensor#ins_pose', 'odom/#sensor#std_odom', 'rawimu/#sensor#raw_imu', 'vehicle_info/#vehicle_info']
        #     pdb.set_trace()
        print(list(freader)[0])  # 第一行 ['frame_id', 'left_lidar', 'middle_lidar', 'right_lidar', 'fuser_lidar', '#cam_10#image_raw#compressed', '#cam_11#image_raw#compressed', '#cam_15#image_raw#compressed', '#cam_2#image_raw#compressed', '#cam_5#image_raw#compressed', 'corr_imu/#sensor#corr_imu', 'ins/#sensor#ins_pose', 'odom/#sensor#std_odom', 'rawimu/#sensor#raw_imu', 'vehicle_info/#vehicle_info']

        fwriter = csv.writer(outcsv, delimiter=",")
        # fwriter.writerow('')

def pandas_read_csv():
    inpath = f's3://tf-shared-data/parse_data/ppl_bag_20211120_143125/timestamp_align_merged.csv'
    out_path = "s3://hhb-zhengchaofan-data-processing-oss/test/test.csv"
    with smart_open(inpath, "r") as incsv, smart_open(out_path, "w",) as outcsv:
        dataframe = pd.read_csv(inpath,f)


if __name__ == '__main__':
    csv_read_csv()