import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold, train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer, roc_curve
import xgboost as xgb


def one_run_cv(X, y, model, k_fold=10):
    n = X.shape[0]
    folds = KFold(n, k_fold, shuffle=True)
    scores = []
    models = []
    tot_score = 0
    i = 1
    for idx_train, idx_test in folds:
        X_train, y_train = X[idx_train], y[idx_train]
        X_test, y_test = X[idx_test], y[idx_test]
        model.fit(X_train, y_train,
                  eval_metric='auc', verbose=True)
        y_prob = model.predict_proba(X_test)[:, 1]
        # print y_prob
        score = roc_auc_score(y_test, y_prob)
        tot_score += score
        scores.append(score)
        models.append(model)
        print 'fold {}: {}'.format(i, score)
        i += 1
    tot_score /= k_fold
    idx = np.argmax(score)
    return tot_score, models[idx]


def one_run_test(X_train, X_test, y_train, y_test):
    xgb1 = xgb.XGBClassifier(learning_rate=0.1, n_estimators=20,
                             max_depth=5, min_child_weight=1,
                             subsample=0.8, colsample_bytree=0.8,
                             objective='binary:logistic',
                             scale_pos_weight=1, seed=27,
                             silent=True)
    score, one_model = one_run_cv(X_train, y_train, xgb1, 3)
    print 'Trainning CV scores: ', score
    y_prob = one_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print 'test auc: ', auc


def tree_nums(X_train, X_test, y_train, y_test):
    trees = [100, 200, 300, 500, 800, 1000]
    scores_train = []
    scores_var = []
    for tree in trees:
        print 'tree numbers: ', tree
        xgb1 = xgb.XGBClassifier(learning_rate=0.1, n_estimators=tree,
                                 max_depth=5, min_child_weight=1,
                                 subsample=0.8, colsample_bytree=0.8,
                                 objective='binary:logistic',
                                 scale_pos_weight=1, seed=27,
                                 silent=True)
        score, one_model = one_run_cv(X_train, y_train, xgb1, 5)
        print 'Trainning CV scores: ', score
        scores_train.append(score)
        y_prob = one_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print 'test auc: ', auc
        scores_var.append(auc)
        print '----------------------'
    plt.figure()
    plt.plot(trees, scores_train, 'b-*')
    plt.plot(trees, scores_var, 'r_.')
    plt.show()


def gamma_nums(X_train, X_test, y_train, y_test):
    gammas = [0, 0.005, 0.01, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
              0.1, 0.15, 0.2]
    scores_train = []
    scores_var = []
    for gamma in gammas:
        print 'gamma numbers: ', gamma
        xgb1 = xgb.XGBClassifier(learning_rate=0.1, n_estimators=100,
                                 max_depth=5, min_child_weight=1,
                                 subsample=0.8, colsample_bytree=0.8,
                                 objective='binary:logistic',
                                 scale_pos_weight=1, seed=27,
                                 nthread=4, gamma=gamma,
                                 silent=True)
        score, one_model = one_run_cv(X_train, y_train, xgb1, 5)
        print 'Trainning CV scores: ', score
        scores_train.append(score)
        y_prob = one_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print 'test auc: ', auc
        scores_var.append(auc)
        print '----------------------'
    plt.figure()
    plt.plot(gammas, scores_train, 'b-*')
    plt.plot(gammas, scores_var, 'r-')
    plt.show()


def eta_nums(X_train, X_test, y_train, y_test):
    lrs = np.arange(0.05, 0.55, 0.05)
    scores_train = []
    scores_var = []
    for lr in lrs:
        print 'learning rate numbers: ', lr
        xgb1 = xgb.XGBClassifier(learning_rate=lr, n_estimators=100,
                                 max_depth=5, min_child_weight=1,
                                 subsample=0.8, colsample_bytree=0.8,
                                 objective='binary:logistic',
                                 scale_pos_weight=1, seed=27,
                                 nthread=4, gamma=0.1,
                                 silent=True)
        score, one_model = one_run_cv(X_train, y_train, xgb1, 5)
        print 'Trainning CV scores: ', score
        scores_train.append(score)
        y_prob = one_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print 'test auc: ', auc
        scores_var.append(auc)
        print '----------------------'
    plt.figure()
    plt.title('cross validation for learning rate')
    plt.plot(lrs, scores_train, 'b-*', label='cv auc')
    plt.plot(lrs, scores_var, 'r-', label='test auc')
    plt.legend()
    plt.show()


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
    plt.show()


def plot_roc_curve(y_true, y_prob):
    tpr, fpr, thretholds = roc_curve(y_true, y_prob)
    area = roc_auc_score(y_true, y_prob)
    x = np.arrange(0, 1, 0.05)
    plt.figure()
    plt.title('ROC curve')
    plt.plot(fpr, tpr, label='AUC = %.2f' % area)
    plt.plot(x, x, 'b--')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('False Possitive Rate')
    plt.ylabel('True Possitive Rate')
    plt.show()


def grid_search_model(X, y, xgb_params, grid_params, k=10):
    xgb_model = xgb.XGBClassifier(xgb_params)
    grid_model = GridSearchCV(xgb_model, param_grid=grid_params,
                              verbose=2, n_jobs=5, iid=False, cv=k,
                              scoring=make_scorer(roc_auc_score))
    grid_model.fit(X, y)
    best_model = grid_model.best_estimator_
    with open('../log/grid_res.txt', 'a+') as f:
        f.write('\n****************************\n')
        f.write('Best Score: %s\n' % (str(grid_model.best_score_)))
        print 'Best Score: ', grid_model.best_score_
        f.write('Best Parms: %s\n' % (str(grid_model.best_params_)))
        print 'Best Model Parameters:\n', grid_model.best_params_
        f.write('------------------------------\n')
        for line in grid_model.grid_scores_:
            f.write(str(line)+'\n')
        f.write('------------------------------\n')
    return best_model, grid_model.grid_scores_


def run_grid_search(X, y, features):
    xgb_params = {
                "n_estimators": 200,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "tree_method": 'exact',
                "nthread": 4,
                # "learning_rate": 0.2,
                # "max_depth": 13,
                "silent": True,
                }
    grid_params = {
                'n_estimators': [100, 300, 500, 700, 900, 1100, 1300, 1500],
                'max_depth': [13],
                'subsample': [0.7],
                'learning_rate': [0.7],
                'colsample_bytree': [1.0],
                'gamma': [0.3],
                'min_child_weight': [1],
                }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    model, grid_scores = grid_search_model(X_train, y_train,
                                           xgb_params, grid_params, 5)
    y_prob = model.predict_proba(X_test)[:, 1]
    test_score = roc_auc_score(y_test, y_prob)
    print 'Final Test Score:', test_score
    # plot_roc_curve(y_test, y_prob)
    # fea_imp = model.feature_importances_
    # plot_feature_importance(fea_imp, features)
    return model, grid_scores


def make_output(model, data):
    X = data.drop(['people_id', 'activity_id'], axis=1).values
    ids = data['activity_id'].values
    y_pred = model.predict(X)
    res = {'activity_id': ids, 'outcome': y_pred}
    pd.DataFrame(res).to_csv('../Data/result.csv', index=False)


def main():
    train_file = '../Data/train.csv'
    test_file = '../Data/test.csv'
    # train_data = pd.read_csv(train_file, parse_dates=['date_p', 'date_a'])
    train_data = pd.read_csv(train_file)
    X = train_data.drop(['outcome', 'people_id', 'activity_id'], axis=1).values
    y = train_data['outcome'].values
    features = train_data.drop(['outcome', 'people_id', 'activity_id'],
                               axis=1).columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # one_run_test(X_train, X_test, y_train, y_test)
    model = run_grid_search(X, y, features)
    sub_data = pd.read_csv(test_file)
    make_output(model, sub_data)


if __name__ == '__main__':
    # main()

    train_file = '../Data/train.csv'
    test_file = '../Data/test.csv'
    # train_data = pd.read_csv(train_file, parse_dates=['date_p', 'date_a'])
    train_data = pd.read_csv(train_file)
    X = train_data.drop(['outcome', 'people_id', 'activity_id'], axis=1).values
    y = train_data['outcome'].values
    features = train_data.drop(['outcome', 'people_id', 'activity_id'],
                               axis=1).columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # one_run_test(X_train, X_test, y_train, y_test)
    model, grid_scores = run_grid_search(X, y, features)
    # sub_data = pd.read_csv(test_file)
    # make_output(model, sub_data)
    # tree_nums(X_train, X_test, y_train, y_test)
    # gamma_nums(X_train, X_test, y_train, y_test)
    # eta_nums(X_train, X_test, y_train, y_test)
