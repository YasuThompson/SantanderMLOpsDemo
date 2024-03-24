import pandas as pd
import numpy as np
import xgboost as xgb
from mapk import mapk
import pickle


from process_data_santander import clean_data, data_engineering, split_data

np.random.seed(2018)



prods = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
             'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
             'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
             'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']


train_path = '../data/santander-product-recommendation/train_ver2_small.csv'
test_path = '../data/santander-product-recommendation/test_ver2.csv'

# use_dates = ['2016-02-28', '2016-03-28', '2016-04-28', '2016-05-28']
use_dates = ['2015-01-28', '2015-02-28', '2015-03-28']
val_dates = ['2015-03-28']
test_dates =  ['2016-06-28']

param = {
        'booster': 'gbtree',
        'max_depth': 8,
        'nthread': 4,
        'num_class': 24,
        'objective': 'multi:softprob',
        'silent': 1,
        'eval_metric': 'mlogloss',
        'eta': 0.1,
        'min_child_weight': 10,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.9,
        'seed': 2018,
        'verbosity': 0
    }

#def validate_with_map7():
#    return 

def main():
    #### Phase 1 ####
    # DADA LAODING. 
    # Preferably in an SQL server or a distributed storage. 
    print("Loading data from csv files")
    trn = pd.read_csv(train_path, sep=',', low_memory=False)
    tst = pd.read_csv(test_path, sep=',',  low_memory=False)
    
    #### Phase 2 ####
    # DATA CLEANING. 
    # Preferably completed before SQL server. 
    print("Cleaninng data.")
    df = clean_data(trn, tst)
    
    #### Phase 3 ####
    # Data engineering. 
    # Preferably implemented with SQL or Apache Spark.
    # Preferably saved to a feature store. 
    print("Labeling data.")
    trn, tst, features = data_engineering(df, use_dates, test_dates)
    
    
    """
    ↑ DATA ENGINEERING
    
    Feature store should be placed here. 
    *A border between data engineering and data scinece. 
    
    ↓ DATA SCIENCE
    """

    
    
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
    