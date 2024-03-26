import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

from process_data_santander import clean_data, data_engineering, split_data
from load_local_data import load_local_data
np.random.seed(2018)


def load_local_data(csv_path):
    return pd.read_csv(csv_path, sep=',', low_memory=False)


def split_data(trn, features, val_dates, prod_cols):
    # 訓練データから新規購買件数だけを抽出します。
    X = []
    Y = []
    for i, prod in enumerate(prod_cols):
        prev = prod + '_prev'
        prX = trn[(trn[prod] == 1) & (trn[prev] == 0)]
        prY = np.zeros(prX.shape[0], dtype=np.int8) + i
        X.append(prX)
        Y.append(prY)
    XY = pd.concat(X)
    Y = np.hstack(Y)
    XY['y'] = Y

    # 訓練、検証データに分離します。

    # XY_trn = XY[XY['fecha_dato'] != vld_date]
    # XY_vld = XY[XY['fecha_dato'] == vld_date]
    XY_trn = XY[~XY['fecha_dato'].isin(val_dates)]
    XY_vld = XY[XY['fecha_dato'].isin(val_dates)]

    # 訓練、検証データを XGBoost 形態に変換します。
    X_trn = XY_trn[features].values
    Y_trn = XY_trn['y'].values
    # dtrn = xgb.DMatrix(X_trn, label=Y_trn, feature_names=features)

    X_vld = XY_vld[features].values
    Y_vld = XY_vld['y'].values
    # dvld = xgb.DMatrix(X_vld, label=Y_vld, feature_names=features)

    return X_trn, Y_trn, X_vld, Y_vld

#def validate_with_map7():
#    return 

def main():
    #### Phase 4 ####
    # DATA SPLIT FOR CROSS VALIDATION. 
    print("Splitting into training and validation data as Numpy arrays..")
    X_trn, Y_trn, X_vld, Y_vld = split_data(trn, features, val_dates)
    
    
    #### Phase 5 ####
    # MODEL INITIALIZATION AND TRAINING. 
    dtrn = xgb.DMatrix(X_trn, label=Y_trn, feature_names=features)
    dvld = xgb.DMatrix(X_vld, label=Y_vld, feature_names=features)
    watch_list = [(dtrn, 'train'), (dvld, 'eval')]
    print("Training a model")
    model = xgb.train(param, dtrn, num_boost_round=10, evals=watch_list, early_stopping_rounds=1)
    
    pickle.dump(model, open("./model/xgb.baseline.pkl", "wb"))
    # best_ntree_limit = model.best_ntree_limit

    #dall = xgb.DMatrix(X_all, label=Y_all, feature_names=features)
    #best_ntree_limit = int(best_ntree_limit * (len(XY_trn) + len(XY_vld)) / len(XY_trn))
    
    print("Feature importance:")
    for kv in sorted([(k,v) for k,v in model.get_fscore().items()], key=lambda kv: kv[1], reverse=True):
        print(kv)
        
        
    #### Phase 5 ####
    # INFERENCE AND EVAULATION    
    print("Inference on test data")    
    X_tst = tst[features].values
    dtst = xgb.DMatrix(X_tst, feature_names=features)
    # preds_tst = model.predict(dtst, ntree_limit=best_ntree_limit)
    preds_tst = model.predict(dtst)
    ncodpers_tst = tst['ncodpers'].values
    #preds_tst = preds_tst - tst.as_matrix(columns=[prod + '_prev' for prod in prods])
    preds_tst = preds_tst - tst[[prod + '_prev' for prod in prods]].values
    
    print("Exporting results on test data")
    submit_file = open('./model/xgb.baseline.2015-06-28', 'w')
    submit_file.write('ncodpers,added_products\n')
    for ncodper, pred in zip(ncodpers_tst, preds_tst):
        y_prods = [(y,p,ip) for y,p,ip in zip(pred, prods, range(len(prods)))]
        y_prods = sorted(y_prods, key=lambda a: a[0], reverse=True)[:7]
        y_prods = [p for y,p,ip in y_prods]
        submit_file.write('{},{}\n'.format(int(ncodper), ' '.join(y_prods)))

    
if __name__ == '__main__':
    
    main()
    