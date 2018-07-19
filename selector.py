# coding=utf-8
import re
import os
import shutil
from collections import OrderedDict
from xml.dom import minidom


class Selector(object):
    def __init__(self, directory, match, file_regs):
        # 开始路径
        self.directory = directory
        # 是否匹配正则
        self.match = match
        # 正则字典
        self.file_regs = file_regs
        # dic_regs带顺序的词典， 字典结构  dict{dict{list}}
        self.dic_regs = self.get_reg_exp()
        # 找到的目标文件路径集合
        self.dic = {}

    def start(self):
        file_list = self.get_files()
        # 文件路径作为key，维度值作为值
        for file in file_list:
            self.parse_xml(file)

    def get_files(self):
        # 获取该路径下所有文件的列表
        filelist = []
        for (root, dirs, files) in os.walk(self.directory):
            temp = [os.path.join(root, files) for files in files]
            filelist.extend(temp)
        return filelist

    # 获取正则表达式
    def get_reg_exp(self):
        dic = OrderedDict()
        with open(self.file_regs, 'r', encoding='utf-8') as f:
            lst = f.readlines()
        for line in lst:
            v = line.strip().split(' ')
            lst_sub = []
            if v[0] in dic.keys():
                if v[1] in dic[v[0]].keys():
                    dic[v[0]][v[1]].append(v[2])
                else:
                    lst_sub.append(v[2])
                    dic[v[0]][v[1]] = lst_sub
            else:
                dic_sub = OrderedDict()
                lst_sub.append(v[2])
                dic_sub[v[1]] = lst_sub
                dic[v[0]] = dic_sub
        return dic

    def parse_xml(self, file):
        try:
            doc = minidom.parse(file)
        except Exception as err:
            self.parse_txt(file)
            return
        root = doc.documentElement

        # 遍历字典 字典结构是  dict{dict{list}},维度{段落{正则}}
        for k_dimension, v in self.dic_regs.items():
            flag = 0
            for k_paragraph, v_regs in v.items():
                for reg in v_regs:
                    lst_paragraph = str(k_paragraph).split('、')
                    for paragraph in lst_paragraph:
                        nodes = root.getElementsByTagName(paragraph)
                        if nodes:
                            txt = nodes[0].childNodes[0].nodeValue
                            flag = is_match(reg, txt)
                        # self.match = True，存入self.dic
                        if flag == 1 and self.match:
                            if k_dimension in self.dic:
                                self.dic[k_dimension].append(file)
                            else:
                                self.dic[k_dimension] = [file]
                            break
                        # self.match = False, 删除文件
                        elif flag == 1 and not self.match:
                            os.remove(file)
                            break
                    if flag == 1:
                        break
                if flag == 1:
                    break

    # 格式错误的xml文档直接转化成字符串处理
    def parse_txt(self, file):
        file_obj = open(file, 'r', encoding='utf-8')
        txt = ''
        try:
            lines = file_obj .readlines()
            for line in lines:
                txt += line
        finally:
            file_obj.close()

        # 遍历字典 字典结构是  dict{dict{list}},维度{段落{正则}}
        for k_dimension, v in self.dic_regs.items():
            flag = 0
            for k_paragraph, v_regs in v.items():
                for reg in v_regs:
                    flag = is_match(reg, txt)
                    if flag == 1 and self.match:
                        if k_dimension in self.dic:
                            self.dic[k_dimension].append(file)
                        else:
                            self.dic[k_dimension] = [file]
                        break
                    # self.match = False, 删除文件
                    elif flag == 1 and not self.match:
                        os.remove(file)
                        break
                if flag == 1:
                    break


# 是否匹配到结果，1 表示匹配到，0表示没有匹配到
def is_match(reg, txt):
    # 这里只通过一个限制条件查找
    pattern = re.compile(reg)
    m = re.search(pattern, str(txt))
    if m:
        return 1
    else:
        return 0


def search(path, match, file_regs):
    with open(file_regs, 'r', encoding='utf-8') as f:
        label = f.readline().split(' ')[0]
    results = []
    dirlist = os.listdir(path)
    count = 0
    for dir_name in dirlist:
        directory = path + '/' + dir_name
        selector = Selector(directory, match, file_regs)
        # 当前路径中的文件总数
        count += len(selector.get_files())
        selector.start()
        if label in selector.dic:
            results.extend(selector.dic[label])
    return label, results, count


def mkdir(dir_path):
    exists = os.path.exists(dir_path)
    if not exists:
        os.makedirs(dir_path)
        print(dir_path + ' 目录创建成功\n')
    else:
        print(dir_path + ' 目录已存在\n')


def copy_files(file_list, target_dir):
    for file in file_list:
        new_file = target_dir + '/' + os.path.basename(file)
        shutil.copyfile(file, new_file)


# 返回结果为字典，字典格式：{文件1：[句子1，句子2。。。]，
#                           。。。}
def find_sentences(files, match, file_regs, label, seq):
    my_dict = {}
    path = 'C:/Users/李亚博/Desktop/同案智推/targets/' + label + '/' + seq + '/'
    for file in files:
        file = path + file
        if file not in my_dict:
            my_dict[file] = find_sentences_in_xml(file, match, file_regs)
    return my_dict


def find_sentences_in_xml(file, match, file_regs):
    results = []
    try:
        doc = minidom.parse(file)
    except Exception as err:
        return find_sentences_in_txt(file, match, file_regs)
    root = doc.documentElement
    selector = Selector(file, match, file_regs)
    # 避免匹配多个正则表达式的语句重复存储
    visited = set()
    for k_dimension, v in selector.dic_regs.items():
        for k_paragraph, v_regs in v.items():
            for reg in v_regs:
                lst_paragraph = str(k_paragraph).split('、')
                for paragraph in lst_paragraph:
                    nodes = root.getElementsByTagName(paragraph)
                    if nodes:
                        txt = nodes[0].childNodes[0].nodeValue
                        sens = txt.split('。')
                        for sen in sens:
                            if is_match(reg, sen) == 1 and sen[0:10] not in visited:
                                results.append('出处段落：' + paragraph + '  出处语句：' + sen + '。')
                                visited.add(sen[0:10])
    return results


# 查找格式错误文档中的匹配语句
def find_sentences_in_txt(file, match, file_regs):
    results = []
    file_obj = open(file, 'r', encoding='utf-8')
    txt = ''
    try:
        lines = file_obj.readlines()
        for line in lines:
            txt += line
    finally:
        file_obj.close()

    selector = Selector(file, match, file_regs)
    visited = set()
    for k_dimension, v in selector.dic_regs.items():
        for k_paragraph, v_regs in v.items():
            for reg in v_regs:
                sens = txt.split('。')
                for sen in sens:
                    if is_match(reg, sen) and sen[0:10] not in visited:
                        results.append('出处语句：' + sen + '。')
                        visited.add(sen[0:10])
    return results


def main():
    current_dir = os.getcwd()
    label_dict = {'1': '事故责任无法认定',
                  '2': '被告主要责任',
                  '3': '被告次要责任',
                  '4': '被告全部责任',
                  '5': '双方平等责任',
                  '6': '被告无责任',
                  '7': '未提起过刑事附带民事诉讼',
                  '8': '受害人有过错'}
    while True:
        option = input('\n案情类别（输入对应数字）：'
                       '\n1 - 事故责任无法认定'
                       '\n2 - 被告主要责任'
                       '\n3 - 被告次要责任'
                       '\n4 - 被告全部责任'
                       '\n5 - 双方平等责任'
                       '\n6 - 被告无责任'
                       '\n7 - 未提起过刑事附带民事诉讼'
                       '\n8 - 受害人有过错'
                       '\n')
        label = label_dict[option]
        flag = input('\n请选择操作：1 - 筛选文件，2 - 生成文档\n')
        path = 'C:/Users/李亚博/Downloads/1'
        seq = os.path.basename(path)

        if int(flag) == 1:
            # 选择是否跳过匹配正则的步骤
            skip = input('\n是否跳过第一步：1 - 否， 2 - 是\n')
            total = 99989
            if int(skip) == 1:
                print('\n正在查找匹配正则的案例，请稍等。。。\n')
                # 搜索匹配正则的案例
                file_regs = current_dir + '/正则字典/' + label + '正则.txt'
                _, file_list, total = search(path, True, file_regs)
                # 检查文件夹是否已经存在
                path = 'C:/Users/李亚博/Desktop/同案智推/targets/' + label
                target_dir = path + '/' + seq
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                # 创建目标文件夹
                mkdir(target_dir)
                # 将匹配的案例文档复制到目标文件夹
                print('\n正在将匹配的文件复制到目标文件夹。。。\n')
                copy_files(file_list, target_dir)

            path = 'C:/Users/李亚博/Desktop/同案智推/targets/' + label
            target_dir = path + '/' + seq

            # 选择是否继续当前操作
            carry_on = input('是否继续筛选：\n1 - 是，2 - 否\n')
            if int(carry_on) == 1:
                file_regs = current_dir + '/正则字典/' + label + '正则_不匹配.txt'
                if os.path.isfile(file_regs):
                    # 删除目标文件夹中不符合要求的案例
                    print('\n正在查找匹配正则的案例，请稍等。。。\n')
                    search(path, False, file_regs)
                else:
                    print(file_regs + ' 不存在\n')

            # 当前目标文件夹中案例数量
            count = len(os.listdir(target_dir))
            print('\n查找完毕，' + str(total) + '个案例中共找到' + str(count) + '个匹配案例\n')

        else:
            # 将匹配的案例与包含的关键语句保存到文档
            file_name = '正则结果/' + label + '.txt'
            file_list = os.listdir('C:/Users/李亚博/Desktop/同案智推/targets/' + label + '/' + seq)
            print('\n正在将结果保存进文档，请稍等。。。\n')
            with open(file_name, 'w', encoding='utf-8') as f:
                # 再次匹配最初的正则
                dic = find_sentences(file_list, True, current_dir + '/正则字典/' + label + '正则.txt', label, seq)
                for file in dic.keys():
                    f.write(str(file)+'\n')
                    for sentence in dic[file]:
                        f.write(str(sentence)+'\n')
                    f.write('\n')
            print('结果已保存到：' + file_name)


if __name__ == "__main__":
    main()
