import os
import shutil
import xml.etree.ElementTree as ET


def copy_files(file_list, src, category):
    dest = 'C:/Users/李亚博/Desktop/同案智推/data/corpus/' + category + '/'
    # 查看目标路径是否存在
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    for file in file_list:
        # 文件完整路径
        file = src + file
        new_file = dest + os.path.basename(file)
        shutil.copyfile(file, new_file)


# 将xml文件转化成txt文件并复制到目标路径
def xml2txt(src, dest):
    category = os.path.basename(src)
    if os.path.exists(dest+category):
        shutil.rmtree(dest+category)
    os.mkdir(dest+category)
    for root, dirs, files in os.walk(src):
        for name in files:
            file_path = os.path.join(root, name)
            txt = get_text(file_path)
            new_path = dest + category + '/' + os.path.splitext(os.path.basename(file_path))[0] + '.txt'
            file_obj = open(new_path, 'w', encoding='utf-8')
            file_obj.write(txt)
            file_obj.close()


def get_text(path):
    txt = ''
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        paragraphs = ['本院认为', '本院查明', '原告诉称', '审理经过']
        for paragraph in paragraphs:
            nodes = root[0].findall(paragraph)
            if nodes:
                txt += nodes[0].text
    except Exception:
        file_obj = open(path, 'r', encoding='utf-8')
        try:
            lines = file_obj.readlines()
            for line in lines:
                txt += line
        finally:
            file_obj.close()
    return txt


def main():

    # category_dict = {'1': '事故责任无法认定',
    #                  '2': '被告主要责任',
    #                  '3': '被告次要责任',
    #                  '4': '被告全部责任',
    #                  '5': '双方平等责任',
    #                  '6': '被告无责任'}

    category_dict = {'1': '被告主要责任',
                     '2': '被告次要责任',
                     '3': '被告全部责任',
                     '4': '双方平等责任',
                     '5': '其他'}

    dest_dir = 'C:/Users/李亚博/Desktop/同案智推/责任构成_1000/'
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.mkdir(dest_dir)
    for category in list(category_dict.values()):
        src_dir = 'C:/Users/李亚博/Desktop/同案智推/' + str(category)
        xml2txt(src_dir, dest_dir)

    while True:

        # user_input = input('\n案情类别（输入对应数字）：'
        #                    '\n1 - 事故责任无法认定'
        #                    '\n2 - 被告主要责任'
        #                    '\n3 - 被告次要责任'
        #                    '\n4 - 被告全部责任'
        #                    '\n5 - 双方平等责任'
        #                    '\n6 - 被告无责任'
        #                    '\n')

        user_input = input('\n案情类别（输入对应数字）：'
                           '\n1 - 被告主要责任'
                           '\n2 - 被告次要责任'
                           '\n3 - 被告全部责任'
                           '\n4 - 双方平等责任'
                           '\n5 - 其他'
                           '\n6 - 退出'
                           '\n')

        if user_input == '6':
            break

        category = category_dict[user_input]
        src_path = 'C:/Users/李亚博/Desktop/同案智推/责任构成_1000/' + category + '/'

        file_list = os.listdir(src_path)
        print('正在复制训练数据集文件，请稍等。。。\n')
        copy_files(file_list, src_path, category)
        print(category + '数据集构造完毕\n')


if __name__ == "__main__":
    main()
