import pickle
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn import metrics


def read_obj(path):
    file_obj = open(path, "rb")
    bunch = pickle.load(file_obj)
    file_obj.close()
    return bunch


def train_rf(tdm, label, seed):
    clf = RandomForestClassifier(random_state=seed)
    clf.fit(tdm, label)
    return clf


def train_xgb(tdm, label, seed):
    # 多分类模型
    clf = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        scale_pos_weight=1,
        random_state=seed
    )
    clf.fit(tdm, label, eval_metric='mlogloss')
    return clf


def train_lgbm(train_set, test_set, seed):
    X_train = train_set.tdm
    y_train = train_set.label
    X_test = test_set.tdm
    y_test = test_set.label
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt',
        num_leaves=31,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed
    )
    clf.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=10,
            verbose=True
            )
    return clf


# 对结果进行投票
def vote(predictions):
    preds = np.transpose(predictions)
    final_result = []
    for pred in preds:
        cnt = Counter(pred)
        label = cnt.most_common(1)[0][0]
        final_result.append(label)
    return final_result


def main():

    # 读取文本向量数据集
    path = "word_bag/tfidfspace.dat"
    bunch_obj = read_obj(path)
    train_set = bunch_obj.train
    test_set = bunch_obj.test

    bags = 10
    seed = 1
    bagged_prediction = []

    for i in range(bags):
        seed += i
        model = train_rf(train_set.tdm, train_set.label, seed)  # Naive-Bayes as base model
        # model = train_xgb(train_set.tdm, train_set.label, seed)  # XGBoost as base model
        # model = train_lgbm(train_set, test_set, seed)  # LightGBM as base model
        pred = model.predict(test_set.tdm)
        bagged_prediction.append(pred)

    y_pred = vote(bagged_prediction)
    y_test = test_set.label
    print(metrics.classification_report(y_test, y_pred, target_names=bunch_obj.target_name))
    print('accuracy: {:.6f}'.format(metrics.accuracy_score(y_test, y_pred)))
    print('\nconfusion matrix:')
    print(metrics.confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
