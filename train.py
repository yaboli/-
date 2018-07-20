import pickle  # 导入持久化类
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC
import xgboost as xgb
import lightgbm as lgb


def readbunchobj(path):  # 读取bunch对象函数
    file_obj = open(path, "rb")
    bunch = pickle.load(file_obj)  # 使用pickle.load反序列化对象
    file_obj.close()
    return bunch


def train_nb(train_set):
    # # alpha越小，迭代次数越多，精度越高
    # clf = MultinomialNB(alpha=0.001).fit(train_set.tdm, train_set.label)

    # 交叉验证
    model = MultinomialNB()
    alphas = [0.01, 0.001, 0.001]
    param_grid = dict(alpha=alphas)
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_log_loss",
        cv=kfold,
        verbose=0
    )
    clf = grid.fit(train_set.tdm, train_set.label)

    # Save trained model
    model_path = 'models/naive_bayes.pkl'
    save_model(model_path, clf)
    return clf


def train_rf(train_set):
    clf = RandomForestClassifier(
        max_depth=5,
        random_state=42)
    clf.fit(train_set.tdm, train_set.label)

    model_path = 'models/random_forest.pkl'
    save_model(model_path, clf)
    return clf


def train_sgd(train_set):
    clf = SGDClassifier(
        loss='hinge',
        penalty='l2',
        alpha=1e-4,
        max_iter=7,
        tol=None,
        random_state=42)
    clf.fit(train_set.tdm, train_set.label)

    # Save trained model
    model_path = 'models/sgd_clf.pkl'
    save_model(model_path, clf)
    return clf


def train_lsvc(train_set):
    # clf = LinearSVC(loss='hinge')
    clf = SVC(C=8,
              kernel='linear',
              decision_function_shape='ovr',
              random_state=42)
    clf.fit(train_set.tdm, train_set.label)

    model_path = 'models/lsvc.pkl'
    save_model(model_path, clf)
    return clf


def train_gbm(train_set):
    clf = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        random_state=10)
    clf.fit(train_set.tdm, train_set.label)

    model_path = 'models/gbm.pkl'
    save_model(model_path, clf)
    return clf


def train_xgb(train_set):
    # 多分类模型
    clf = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=3,
        min_child_weight=1,
        gamma=0.4,
        subsample=0.7,
        colsample_bytree=0.6,
        objective='multi:softprob',
        scale_pos_weight=1,
        random_state=27
    )
    clf.fit(train_set.tdm, train_set.label, eval_metric='mlogloss')

    model_path = 'models/xgboost.pkl'
    save_model(model_path, clf)
    return clf


def train_lgbm(train_set, test_set):
    X_train = train_set.tdm
    y_train = train_set.label
    X_test = test_set.tdm
    y_test = test_set.label
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt',
        num_leaves=31,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8)
    clf.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=10,
            verbose=True
            )

    model_path = 'models/lightgbm.pkl'
    save_model(model_path, clf)
    return clf


def save_model(model_path, model):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def main():
    path = "word_bag/tfidfspace.dat"
    bunch_obj = readbunchobj(path)  # 导入训练集向量空间
    train_set = bunch_obj.train
    test_set = bunch_obj.test

    print(train_set.tdm.shape)

    while True:
        user_input = input(
            '\n输入模型类型：'
            '\n1 - Naive Bayes'
            '\n2 - Random Forest'
            '\n3 - SGD Classifier'
            '\n4 - Linear SVC'
            '\n5 - GBM'
            '\n6 - XGBoost'
            '\n7 - LightGBM'
            '\n8 - 退出'
            '\n')

        if user_input == '8':
            break

        print('\nFitting model with training set...')

        if user_input == '1':
            clf = train_nb(train_set)
        elif user_input == '2':
            clf = train_rf(train_set)
        elif user_input == '3':
            clf = train_sgd(train_set)
        elif user_input == '4':
            clf = train_lsvc(train_set)
        elif user_input == '5':
            clf = train_gbm(train_set)
        elif user_input == '6':
            clf = train_xgb(train_set)
        elif user_input == '7':
            clf = train_lgbm(train_set, test_set)

        print('\nFitting completed')

        # 预测分类结果
        actual = test_set.label
        predicted = clf.predict(test_set.tdm)
        print(metrics.classification_report(actual, predicted, target_names=bunch_obj.target_name))
        print('accuracy: {:.6f}'.format(metrics.accuracy_score(actual, predicted)))


if __name__ == "__main__":
    main()
