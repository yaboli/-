import pickle
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import time


def readbunchobj(path):  # 读取bunch对象函数
    file_obj = open(path, "rb")
    bunch = pickle.load(file_obj)  # 使用pickle.load反序列化对象
    file_obj.close()
    return bunch


def save_model(model_path, model):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def main():
    # 开始计时
    start = time.time()

    # 读取并分割训练与测试数据集
    path = 'word_bag/tfidfspace.dat'
    bunch_obj = readbunchobj(path)
    X_train = bunch_obj.train.tdm
    y_train = bunch_obj.train.label
    X_test = bunch_obj.test.tdm
    y_test = bunch_obj.test.label

    model_path = 'models/xgb_best.pkl'

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
        random_state=27)

    num_splits = 5

    # -----------------------调参--------------------------

    # # 1. max_depth和min_child_weight
    # # Best: -0.374535 using {'max_depth': 3, 'min_child_weight': 1}
    # max_depth = range(3, 10, 2)
    # min_child_weight = range(1, 6, 2)
    # param_grid = dict(max_depth=max_depth,
    #                   min_child_weight=min_child_weight)
    # num_fits = len(max_depth) * len(min_child_weight) * num_splits

    # # 2. gamma
    # # Best: -0.366734 using {'gamma': 0.4}
    # gamma = [i / 10.0 for i in range(0, 5)]
    # param_grid = dict(gamma=gamma)
    # num_fits = len(gamma) * num_splits

    # 3. subsample和colsample_bytree
    # Best: -0.358201 using {'colsample_bytree': 0.6, 'subsample': 0.7}
    subsample = [i / 10.0 for i in range(6, 10)]
    colsample_bytree = [i / 10.0 for i in range(6, 10)]
    param_grid = dict(subsample=subsample,
                      colsample_bytree=colsample_bytree)
    num_fits = len(subsample) * len(colsample_bytree) * num_splits

    # -----------------------交叉验证--------------------------
    kfold = StratifiedKFold(n_splits=num_splits, shuffle=True)
    grid_search = GridSearchCV(clf,
                               param_grid,
                               scoring='neg_log_loss',
                               cv=kfold,
                               verbose=num_fits)
    clf = grid_search.fit(X_train, y_train)

    save_model(model_path, clf)

    print("Best: %f using %s" % (clf.best_score_, clf.best_params_))
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    params = clf.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    clf = pickle.load(open(model_path, 'rb'))
    y_pred = clf.predict(X_test)
    print(metrics.classification_report(y_test, y_pred, target_names=bunch_obj.target_name))
    print('accuracy: {:.6f}'.format(metrics.accuracy_score(y_test, y_pred)))

    # 停止计时
    end = time.time()
    print('\nTotal time : {:.2f} {}'.format((end - start) / 60, 'minutes'))


if __name__ == '__main__':
    main()
