import os

from refile import smart_listdir


def step1():
    root = 's3://tf-shared-data/parse_data/'
    for name in smart_listdir(root):
        try:
            date = name.split('_')[2]
        except:
            continue
        if '3dbmk' in name or 'ppl_bag' not in name:
            continue
        if date < '20220116' or date > '20220118':
            continue
        cmd = f'oss rm --rec {os.path.join(root, name)}'
        print(cmd)


if __name__ == '__main__':
    step1()
