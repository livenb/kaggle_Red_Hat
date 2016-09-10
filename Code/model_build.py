import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold, train_test_split
from sklearn.grid_search import GridSearchCV
from sklean.metrics import roc_auc_score
import xgboost as xgb


def one_run_cv(X, y):
    xgb1 = xgb.XGBClassifier(
                     learning_rate =0.1,
                        n_estimators=1000,
                        max_depth=5,
                    min_child_weight=1,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 scale_pos_weight=1,
 seed=27)

def grid_search_model(X, y, xgb_params, grid_params):
   xgb_model = xgb.XGBClassifier(xgb_params)
   xgb.XGBClassifier(self, max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='binary:logistic', nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, seed=0, missing=None)
    

if __name__ == '__main__':
    train_file = '../Data/train.csv'
    test_file = '../Data/test.csv'
    train_data = pd.read_csv(train_file, parse_dates=['date_p', 'date_a'])
    label = 'outcome'
    X = train_data.drop(['outcome', 'people_id', 'activity_id'], axis=1).values
    y = train_data['outcome'].values
    features = train_data.drop(['outcome', 'people_id', 'activity_id'], axis=1).columns
    xgb_params = {
                "objective": "binary:logistic",
                "booster": "gbtree",
                "eval_metric": "auc",
                "tree_method": 'exact',
                "silent": 1,
                }
    grid_params = {
                'num_boost_round': [100, 250, 500],
                'eta': [0.05, 0.1, 0.3],
                'max_depth': [6, 9, 12],
                'subsample': [0.9, 1.0],
                'colsample_bytree': [0.9, 1.0],
                }
