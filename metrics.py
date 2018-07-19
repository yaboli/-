import pickle
from sklearn import metrics


def readbunchobj(path):  # 读取bunch对象函数
    file_obj = open(path, "rb")
    bunch = pickle.load(file_obj)  # 使用pickle.load反序列化对象
    file_obj.close()
    return bunch


def evaluate(model_path):
    clf = pickle.load(open(model_path, 'rb'))
    path = "word_bag/tfidfspace.dat"
    bunch_obj = readbunchobj(path)
    test_set = bunch_obj.test
    actual = test_set.label
    predicted = clf.predict(test_set.tdm)
    print(metrics.classification_report(actual, predicted, target_names=bunch_obj.target_name))
    print('accuracy: {:.6f}'.format(metrics.accuracy_score(actual, predicted)))
    print('\nconfusion matrix:')
    print(metrics.confusion_matrix(actual, predicted))


def main():

    model_path_dict = {'1': 'models/naive_bayes.pkl',
                       '2': 'models/random_forest.pkl',
                       '3': 'models/sgd_clf.pkl',
                       '4': 'models/lsvc.pkl',
                       '5': 'models/gbm.pkl',
                       '6': 'models/xgboost.pkl',
                       '7': 'models/lightgbm.pkl'}

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
        if user_input == str(len(model_path_dict.keys())+1):
            break
        model_path = model_path_dict[user_input]
        evaluate(model_path)


if __name__ == '__main__':
    main()
