import re
import os
import shutil
import xml.etree.ElementTree as ET


def parse_xml(file, reg, dir1, dir2):
    try:
        tree = ET.parse(file)
    except Exception as err:
        parse_txt(file, reg, dir1, dir2)
        return
    root = tree.getroot()
    paragraphs = ['本院认为', '本院查明', '审理经过', '原告诉称']
    flag = 0
    for paragraph in paragraphs:
        nodes = root[0].findall(paragraph)
        if nodes:
            txt = nodes[0].text
            flag = is_match(reg, txt)
            if flag == 1:
                new_file = dir1 + os.path.basename(file)
                shutil.copyfile(file, new_file)
                break
    if flag == 0:
        new_file = dir2 + os.path.basename(file)
        shutil.copyfile(file, new_file)
    return


def parse_txt(file, reg, dir1, dir2):
    file_obj = open(file, 'r', encoding='utf-8')
    txt = ''
    try:
        lines = file_obj.readlines()
        for line in lines:
            txt += line
    finally:
        file_obj.close()
    flag = is_match(reg, txt)
    if flag == 1:
        new_file = dir1 + os.path.basename(file)
        shutil.copyfile(file, new_file)
    else:
        new_file = dir2 + os.path.basename(file)
        shutil.copyfile(file, new_file)


def is_match(reg, txt):
    # 这里只通过一个限制条件查找
    pattern = re.compile(reg)
    m = re.search(pattern, str(txt))
    if m:
        return 1
    else:
        return 0


def main():
    directory = 'C:/Users/李亚博/Downloads/其他'

    categorty = '双方平等责任'

    dir1 = directory + '/' + categorty + '/'
    dir2 = directory + '/非' + categorty + '/'
    if os.path.exists(dir1):
        shutil.rmtree(dir1)
    if os.path.exists(dir2):
        shutil.rmtree(dir2)
    os.makedirs(dir1)
    os.makedirs(dir2)

    file_list = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            file_list.append(os.path.join(root, name))

    reg = '(同等|平等|相同|50%).{0,10}责任'
    for file in file_list:
        parse_xml(file, reg, dir1, dir2)


if __name__ == "__main__":
    main()
