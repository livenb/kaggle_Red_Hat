import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import roc_auc_score, make_scorer, roc_curve
from tqdm import tqdm
import xgboost as xgb


def one_run_cv(X, y, model, k_fold=10):
    n = X.shape[0]
    folds = KFold(n, k_fold, shuffle=True)
    scores = []
    models = []
    tot_score = 0
    pbar = tqdm(total=k_fold)
    i = 1
    with open('../log/result.log', 'a+') as f:
        f.write('**********************\n')
        f.write('The result of with parameters:\n')
        f.write("learning_rate=0.7, n_estimators=1000, gamma=0.3, max_depth=14, min_child_weight=1,subsample=0.6, colsample_bytree=1.0, objective='binary:logistic', scale_pos_weight=1")
        for idx_train, idx_test in folds:
            X_train, y_train = X[idx_train], y[idx_train]
            X_test, y_test = X[idx_test], y[idx_test]
            model.fit(X_train, y_train,
                      eval_metric='auc', 
                      # early_stopping_rounds=50,
                      verbose=True)
            y_prob = model.predict_proba(X_test)[:, 1]
            # print y_prob
            score = roc_auc_score(y_test, y_prob)
            tot_score += score
            scores.append(score)
            models.append(model)
            pbar.update(1)
            f.write('\nfold {}: {}\n'.format(i, score))
            i += 1
        f.write('------------------------\n')
        tot_score /= k_fold
        f.write('The auc score from training:\t{}'.format(tot_score))
    idx = np.argmax(score)
    return tot_score, models[idx]


def one_run_test(X_train, X_test, y_train, y_test):
    xgb1 = xgb.XGBClassifier(learning_rate=0.7, n_estimators=1000,
                             gamma=0.3,
                             max_depth=14, min_child_weight=1,
                             subsample=0.6, colsample_bytree=1.0,
                             objective='binary:logistic',
                             scale_pos_weight=1)
    score, one_model = one_run_cv(X_train, y_train, xgb1, 10)
    y_prob = one_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    with open('../log/result.log', 'a+') as f:
        f.write('\n---------------------\n')
        f.write('The auc score from test:\t{}'.format(auc))
    print 'test auc: ', auc
    return one_model


def plot_feature_importance(fea_imp, features, num_fea=None):
    if num_fea:
        k = num_fea
    else:
        k = len(fea_imp)
    idx = fea_imp.argsort()[::-1]
    plt.figure()
    plt.title('Feature Importance')
    plt.bar(range(k), fea_imp[idx][:k], align="center")
    plt.xticks(range(k), features[idx][:k])
    plt.savefig('../Img/fea_imp.png', dpi=300)


def plot_roc_curve(y_true, y_prob):
    tpr, fpr, thretholds = roc_curve(y_true, y_prob)
    area = roc_auc_score(y_true, y_prob)
    x = np.arange(0, 1, 0.05)
    plt.figure()
    plt.title('ROC curve')
    plt.plot(fpr, tpr, label='AUC = %.2f' % area)
    plt.plot(x, x, 'b--')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('False Possitive Rate')
    plt.ylabel('True Possitive Rate')
    plt.savefig('../Img/roc.png', dpi=300)


def make_output(model, data):
    X = data.drop(['people_id', 'activity_id'], axis=1).values
    ids = data['activity_id'].values
    y_pred = model.predict(X)
    res = {'activity_id': ids, 'outcome': y_pred}
    pd.DataFrame(res).to_csv('../Data/result.csv', index=False)


if __name__ == '__main__':
    train_file = '../Data/train.csv'
    test_file = '../Data/test.csv'
    train_data = pd.read_csv(train_file)
    X = train_data.drop(['outcome', 'people_id', 'activity_id'], axis=1).values
    y = train_data['outcome'].values
    features = train_data.drop(['outcome', 'people_id', 'activity_id'],
                               axis=1).columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = one_run_test(X_train, X_test, y_train, y_test)
    fea_imp = model.plot_feature_importances_
    plot_feature_importance(fea_imp, features, num_fea=10)
