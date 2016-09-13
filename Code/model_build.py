import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold, train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
import xgboost as xgb


def one_run_cv(X, y, model, k_fold=10):
    n = X.shape[0]
    folds = KFold(n, k_fold, shuffle=True)
    scores = []
    models = []
    tot_score = 0
    i = 0
    for idx_train, idx_test in folds:
        print 'fold: ', i
        i += 1
        X_train, y_train = X[idx_train], y[idx_train]
        X_test, y_test = X[idx_test], y[idx_test]
        model.fit(X_train, y_train, eval_metric='auc', verbose=True)
        y_prob = model.predict_proba(X_test)[:, 1]
        # print y_prob
        score = roc_auc_score(y_test, y_prob)
        tot_score += score
        scores.append(score)
        models.append(model)
        print score
    tot_score /= k_fold
    idx = np.argmax(score)
    return tot_score, models[idx]


def one_run_test(X_train, X_test, y_train, y_test):
    xgb1 = xgb.XGBClassifier(learning_rate=0.1, n_estimators=500,
                             max_depth=5, min_child_weight=1,
                             subsample=0.8, colsample_bytree=0.8,
                             objective='binary:logistic',
                             scale_pos_weight=1, seed=27,
                             silent=True)
    score, one_model = one_run_cv(X_train, y_train, xgb1)
    print 'Trainning CV scores: ', score
    y_prob = one_model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_prob)
    print 'test auc: ', auc


def grid_search_model(X, y, xgb_params, grid_params):
    xgb_model = xgb.XGBClassifier(xgb_params)
    grid_model = GridSearchCV(xgb_model, param_grid=grid_params,
                              verbose=1, n_jobs=-1, iid=False, cv=10,
                              scoring=make_scorer(roc_auc_score))
    grid_model.fit(X, y)
    best_model = grid_model.best_estimator_
    print 'Best Score: ', grid_model.best_score_
    print 'Best Model Parameters:\n', grid_model.best_params_
    return best_model, grid_model.grid_scores_


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


def run_grid_search(X, y, features):
    xgb_params = {
                "n_estimators": 1000,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "tree_method": 'exact',
                "silent": True,
                }
    grid_params = {
                'gamma': [0, 0.01, 0.05, 0.08, 0.1, 0.15, 0.2],
                'learning_rate': np.arange(0.05, 0.55, 0.05),
                'max_depth': range(3, 12, 2),
                'min_child_weight': range(1, 10),
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model, grid_scores = grid_search_model(X_train, y_train,
                                           xgb_params, grid_params)
    y_prob = model.predict_proba(X_test)
    test_score = roc_auc_score(y_test, y_prob)
    print 'Final Test Score:', test_score
    fea_imp = model.feature_importances_
    plot_feature_importance(fea_imp, features)
    return model


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
    one_run_test(X_train, X_test, y_train, y_test)
    model = run_grid_search(X, y, features)
    sub_data = pd.read_csv(test_file)
    make_output(model, sub_data)


if __name__ == '__main__':
    main()
