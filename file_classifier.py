import os
import jieba
import pickle
import pandas as pd
import xml.etree.ElementTree as ET


def read_obj(path):
    file_obj = open(path, "rb")
    obj = pickle.load(file_obj)
    file_obj.close()
    return obj


def get_file_list(path):
    file_list = []
    for root, _, file_names in os.walk(path):
        for file_name in file_names:
            file_list.append(os.path.join(root, file_name))
    return file_list


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


def cut_words(txt):
    return " ".join(jieba.cut(txt))


def main():
    jieba.load_userdict('分词词典/solr分词--包含法律词汇.txt')

    root_path = 'C:/Users/李亚博/Downloads/random_100000'
    dir_list = []
    for root, dirs, _ in os.walk(root_path):
        for dir in dirs:
            dir_list.append(os.path.join(root, dir))

    vect_path = 'word_bag/vectorizer.pkl'
    vectorizer = read_obj(vect_path)

    model_path = 'models/xgboost.pkl'
    clf = read_obj(model_path)

    file_paths = []
    predictions = []

    count = 0
    stop = 10000

    for dir in dir_list:

        file_list = get_file_list(dir)
        contents = []

        for file_path in file_list:
            txt = get_text(file_path)
            segments = cut_words(txt)
            contents.append(segments)

        tdm = vectorizer.transform(contents).toarray()
        y_pred = clf.predict(tdm)

        file_paths.extend(file_list)
        predictions.extend(y_pred)

        count += 1
        if count >= stop:
            break

    d = {'文件路径': file_paths, '标签': predictions}
    df = pd.DataFrame(data=d)
    file_name = 'results.xlsx'
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()


if __name__ == '__main__':
    main()
