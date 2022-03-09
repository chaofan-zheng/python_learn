import re


def solution(paragragh):
    paragragh = paragragh.strip()
    paragragh+=' '
    res_dict = {}
    all_words = re.findall(r'([A-Z]|[a-z].*) ', paragragh)
    print(all_words)
    for word in all_words:
        word = word.lower()
        count = res_dict.setdefault(word,1)
        count+=1
        res_dict[word] = count


def solution2(paragragh:str,banned:list):
    paragragh = paragragh.strip()
    res_dict = {}
    all_words = paragragh.split(' ')
    print(all_words)
    for word in all_words:
        word = word.lower()
        # if word.endswith(''):
        word = re.findall(r'[a-z]*',word)[0]
        count = res_dict.setdefault(word,1)
        count+=1
        res_dict[word] = count
    sort_list=[]
    for word,count in res_dict.items():
        if word in banned:
            continue
        sort_list.append({'word':word,'count':count})
    sort_list.sort(key=lambda x:x['count'])
    return sort_list[-1]['word']




if __name__ == '__main__':
    paragragh = 'Bob hit a ball, the hit BALL flew it was hit.'
    res = solution2(paragragh,banned=['hit'])
    print(res)
