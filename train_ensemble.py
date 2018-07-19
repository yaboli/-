import pickle
import xgboost as xgb
from sklearn import metrics
import numpy as np


def readbunchobj(path):  # 读取bunch对象函数
    file_obj = open(path, "rb")
    bunch = pickle.load(file_obj)  # 使用pickle.load反序列化对象
    file_obj.close()
    return bunch


def main():
    model_path_1 = 'models/gbm.pkl'
    model_path_2 = 'models/random_forest.pkl'
    model_path_3 = 'models/naive_bayes.pkl'

    # 读取并分割训练与测试数据集
    path = 'word_bag/tfidfspace.dat'
    bunch_obj = readbunchobj(path)
    train_set = bunch_obj.train
    test_set = bunch_obj.test
    X_train = train_set.tdm
    y_train = train_set.label
    X_test = test_set.tdm
    y_test = test_set.label

    # 载入训练好的模型
    clf1 = pickle.load(open(model_path_1, 'rb'))
    clf2 = pickle.load(open(model_path_2, 'rb'))
    clf3 = pickle.load(open(model_path_3, 'rb'))

    # 第一阶段：使用载入的模型对数据进行预测
    y_train_1 = clf1.predict_proba(X_train)
    y_train_2 = clf2.predict_proba(X_train)
    y_train_3 = clf3.predict_proba(X_train)
    y_test_1 = clf1.predict_proba(X_test)
    y_test_2 = clf2.predict_proba(X_test)
    y_test_3 = clf3.predict_proba(X_test)

    # 将模型预测结果合并
    X_train = np.concatenate((np.concatenate((y_train_1, y_train_2), axis=1), y_train_3), axis=1)
    X_test = np.concatenate((np.concatenate((y_test_1, y_test_2), axis=1), y_test_3), axis=1)

    # 第二阶段：将第一阶段的结果作为训练数据继续训练

    # 多分类模型
    clf = xgb.XGBClassifier(learning_rate=0.01,
                            n_estimators=100,
                            max_depth=5,
                            min_child_weight=3,
                            gamma=0.2,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            objective='multi:softprob',
                            # random_state=27
                            )
    clf.fit(X_train, y_train, eval_metric='mlogloss')

    model_path = 'models/ensemble.pkl'
    pickle.dump(clf, open(model_path, 'wb'))

    y_pred = clf.predict(X_test)
    print(metrics.classification_report(y_test, y_pred, target_names=bunch_obj.target_name))
    print('accuracy: {:.6f}'.format(metrics.accuracy_score(y_test, y_pred)))
    print('\nconfusion matrix:')
    print(metrics.confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    main()
