import re
import os


def precision(path, category, rest, reg):
    file_list = os.listdir(path+category)
    tp = 0
    # 遍历本目录中的文档并计算出true positive
    for file in file_list:
        file = path + category + '/' + file
        file_obj = open(file, 'r', encoding='utf-8')
        txt = file_obj.read()
        if is_match(reg, txt) == 1:
            tp += 1
        file_obj.close()

    fp = 0
    for cate in rest:
        file_list = os.listdir(path+cate)
        for file in file_list:
            file = path + cate + '/' + file
            file_obj = open(file, 'r', encoding='utf-8')
            txt = file_obj.read()
            if is_match(reg, txt) == 1:
                fp += 1
            file_obj.close()

    result = tp / (tp + fp)
    return result


def recall(path, category, reg):
    file_list = os.listdir(path+category)
    total = len(file_list)
    count = 0
    for file in file_list:
        file = path + category + '/' + file
        file_obj = open(file, 'r', encoding='utf-8')
        txt = file_obj.read()
        if is_match(reg, txt) == 1:
            count += 1
        file_obj.close()
    result = count / total
    return result, count


def is_match(reg, txt):
    pattern = re.compile(reg)
    m = re.search(pattern, str(txt))
    if m:
        return 1
    else:
        return 0


def main():
    categories = ['被告主要责任', '被告全部责任', '被告次要责任', '双方平等责任']
    path = 'C:/Users/李亚博/Desktop/同案智推/责任构成临时/'
    iter = 0
    regs = ['(二人|两人|原被告|双方|共同).{0,30}(承担|负|应).{0,20}(同等|平等|相同|50%).{0,10}责任',
            '被告.{0,7}(承担|负).{0,5}(次要|[1-4]0%)(的)?.{0,5}责任',
            '(承担|负).{0,15}(全部责任|全责)',
            '(承担|负).{0,15}主要责任']
    tp_total = 0
    total = 0
    precisions = []
    recalls = []
    while iter < 4:
        category = categories.pop()  # 取出categories中最后一个元素
        # print(categories)
        # print(category)
        reg = regs[iter]
        # print(reg)
        precision_score = precision(path, category, categories, reg)
        precisions.append(precision_score)
        recall_score, tp = recall(path, category, reg)
        # print(tp)
        recalls.append(recall_score)
        tp_total += tp
        count = len(os.listdir(path+category))
        # print(count)
        total += count
        categories = [category] + categories  # 将之前取出的元素放置序列顶部
        # print(categories)
        iter += 1
        # print()

    categories.reverse()
    print(categories)
    print('精度：')
    print(["{:.6f}".format(i) for i in precisions])
    print('召回率：')
    print(["{:.6f}".format(i) for i in recalls])
    print('F1-score：')

    # 计算f1 scores
    f1_scores = []
    for p, r in zip(precisions, recalls):
        f1_score = 2 * (p * r) / (p + r)
        f1_scores.append(f1_score)
    print(["{:.6f}".format(i) for i in f1_scores])

    accuracy = tp_total / total
    print("准确率: {:.6f}".format(accuracy))


if __name__ == '__main__':
    main()
