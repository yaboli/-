import pickle
from sklearn.utils import shuffle
from sklearn.datasets.base import Bunch
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import xgboost as xgb
import numpy as np


def read_obj(path):
    file_obj = open(path, "rb")
    bunch = pickle.load(file_obj)
    file_obj.close()
    return bunch


def build_subsets(train_set):
    tdm = train_set.tdm
    label = train_set.label
    tdm, label = shuffle(tdm, label)
    split_ratio = 0.5
    split_index = int(len(label) * split_ratio)
    set_A = Bunch(tdm=tdm[:split_index], label=label[:split_index])
    set_B = Bunch(tdm=tdm[split_index:], label=label[split_index:])
    return set_A, set_B


def train_rf(tdm, label):
    clf = RandomForestClassifier(max_depth=5)
    clf.fit(tdm, label)
    return clf


def train_nb(tdm, label):
    clf = MultinomialNB(alpha=0.001)
    clf.fit(tdm, label)
    return clf


def train_gbm(tdm, label):
    clf = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5)
    clf.fit(tdm, label)
    return clf


def train_xgb(tdm, label):
    # 多分类模型
    clf = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=5,
        min_child_weight=1,
        # gamma=0.4,
        # subsample=0.7,
        # colsample_bytree=0.6,
        objective='multi:softprob',
        scale_pos_weight=1,
        # random_state=27
    )
    clf.fit(tdm, label, eval_metric='mlogloss')
    return clf


def main():

    # 以下代码参照了Coursera课程中关于Stacking的介绍
    # 课程名称：How to Win a Data Science Competition: Learn from Top Kagglers
    # 链接：https://www.coursera.org/lecture/competitive-data-science/stacking-Qdtt6

    # 读取文本向量数据集
    path = "word_bag/tfidfspace.dat"
    bunch_obj = read_obj(path)
    train_set = bunch_obj.train
    test_set = bunch_obj.test

    # 构建训练集与验证集
    set_A, set_B = build_subsets(train_set)
    set_C = Bunch(tdm=test_set.tdm, label=test_set.label)

    # 训练基础模型，然后使用验证集B作为输入预测结果
    clf_0 = train_rf(set_A.tdm, set_A.label)
    pred_0 = clf_0.predict_proba(set_B.tdm)
    clf_1 = train_nb(set_A.tdm, set_A.label)
    pred_1 = clf_1.predict_proba(set_B.tdm)
    clf_2 = train_gbm(set_A.tdm, set_A.label)
    pred_2 = clf_2.predict_proba(set_B.tdm)

    # 利用验证集预测出的结果构建矩阵B1
    set_B1 = np.concatenate((pred_0, pred_1, pred_2), axis=1)
    # 以B1作为训练集训练meta模型
    clf = train_xgb(set_B1, set_B.label)

    # 使用基础模型对测试集C进行预测
    pred_0 = clf_0.predict_proba(set_C.tdm)
    pred_1 = clf_1.predict_proba(set_C.tdm)
    pred_2 = clf_2.predict_proba(set_C.tdm)
    # 使用C的预测结果构建矩阵C1
    set_C1 = np.concatenate((pred_0, pred_1, pred_2), axis=1)

    y_pred = clf.predict(set_C1)
    y_test = set_C.label
    print(metrics.classification_report(y_test, y_pred, target_names=bunch_obj.target_name))
    print('accuracy: {:.6f}'.format(metrics.accuracy_score(y_test, y_pred)))
    print('\nconfusion matrix:')
    print(metrics.confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    main()
