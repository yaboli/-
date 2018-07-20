import os
import jieba
from sklearn.datasets.base import Bunch
import pickle


# 定义两个函数，用于读取和保存文件

def savefile(path, content):  # 定义一个用于保存文件的函数
    fp = open(path, "w", encoding='utf-8')
    fp.write(content)
    fp.close()


def readfile(path):  # 定义一个用于读取文件的函数
    fp = open(path, "r", encoding='utf-8')
    content = fp.read()
    fp.close()
    return content  # 函数返回读取的内容


# 以下是整个语料库的分词主程序
def cut_words():
    path = "C:/Users/李亚博/Desktop/同案智推/data/"
    corpus_path = path + "corpus/"  # 未分词分类语料库路径
    seg_path = path + "seg/"  # 分词后分类语料库路径

    catelist = os.listdir(corpus_path)  # os.listdir获取cor_path下的所有子目录

    for mydir in catelist:  # 遍历所有子目录
        class_path = corpus_path + mydir + "/"  # 构造分类子目录的路径
        seg_dir = seg_path + mydir + "/"  # 构造分词后的语料分类目录

        if not os.path.exists(seg_dir):  # 是否存在目录，如果没有则创建
            os.makedirs(seg_dir)

        file_list = os.listdir(class_path)  # 获取目录下的所有文件

        for file_path in file_list:  # 遍历目录下的所有文件
            fullname = class_path + file_path  # 文件路径
            content = readfile(fullname).strip()  # 读取文件，strip()用于移除字符串头尾指定的字符，即移除头尾的空格
            content = content.replace("\r\n", "").strip()  # 将空格和换行替代为无
            content_seg = jieba.cut(content)  # 利用jieba分词
            # content_seg = jieba.cut(content, HMM=False)  # 利用jieba分词

            # 调用函数保存文件，保存路径为：seg_dir+file_path,用空格将分词后的词连接起来
            savefile(seg_dir + file_path, " ".join(content_seg))

    print("中文语料分词结束")


#############################################################################

def generate_dat_file():
    # 为了便于后续的向量空间模型的生成，分词后的文本还要转换为文本向量信息并对象化
    # 引入Scikit-Learn的Bunch类

    bunch = Bunch(target_name=[], label=[], filename=[], contents=[])

    # Bunch类提供键值对的对象形式
    # target_name:所有分类集名称列表
    # label:每个文件的分类标签列表
    # filename：文件路径
    # contents：分词后的文件词向量形式

    wordbag_path = "word_bag/"
    if not os.path.exists(wordbag_path):
        os.makedirs(wordbag_path)
    wordbag_path += "train_set.dat"  # 分词语料Bunch对象持久化文件路径

    path = "C:/Users/李亚博/Desktop/同案智推/data/"
    seg_path = path + "seg/"  # 分词后分类语料库路径
    catelist = os.listdir(seg_path)  # 获取分词后语料库的所有子目录（子目录名是类别名）
    bunch.target_name.extend(catelist)  # 将所有类别信息保存到Bunch对象

    for mydir in catelist:  # 遍历所有子目录
        class_path = seg_path + mydir + "/"  # 构造子目录路径
        file_list = os.listdir(class_path)  # 获取子目录内的所有文件
        for file_path in file_list:  # 遍历目录内所有文件
            fullname = class_path + file_path  # 构造文件路径
            bunch.label.append(mydir)  # 保存当前文件的分类标签（mydir为子目录即类别名）
            bunch.filename.append(fullname)  # 保存当前文件的文件路径（full_name为文件路径）
            bunch.contents.append(readfile(fullname).strip())  # 保存文件词向量（调用readfile函数读取文件内容）

    file_obj = open(wordbag_path, "wb")  # 打开前面构造的持久化文件的路径，准备写入
    pickle.dump(bunch, file_obj)  # pickle模块持久化信息，bunch是要持久化的文件，已添加了信息。file_obj是路径
    file_obj.close()
    # 　之所以要持久化，类似游戏中途存档，分词后，划分一个阶段，将分词好的文件存档，后面再运行就不用重复分词了

    print("构建文本对象结束！！")

    # 持久化后生成一个train_set.dat文件，保存着所有训练集文件的所有分类信息
    # 保存着每个文件的文件名，文件所属分类和词向量


def main():
    jieba.load_userdict('分词词典/solr分词--包含法律词汇.txt')
    cut_words()
    generate_dat_file()


if __name__ == "__main__":
    main()
