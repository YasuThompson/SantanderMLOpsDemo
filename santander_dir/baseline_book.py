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

categorical_cols = ['ind_empleado', 'pais_residencia', 'sexo', 'tiprel_1mes', 'indresi', 'indext', 'conyuemp', 'canal_entrada', 'indfall', 'tipodom', 'nomprov', 'segmento']

train_path = '../data/santander/train_ver2.csv'
test_path = '../data/santander/test_ver2.csv'

# use_dates = ['2015-01-28', '2015-02-28', '2015-03-28', '2015-04-28',
#        '2015-05-28', '2015-06-28', '2015-07-28', '2015-08-28',
#        '2015-09-28', '2015-10-28', '2015-11-28', '2015-12-28',
#        '2016-01-28', '2016-02-28', '2016-03-28', '2016-04-28',
#        '2016-05-28']

# test_date =  '2016-06-28'


use_dates = ['2016-02-28', '2016-03-28', '2016-04-28', '2016-05-28']
test_date =  '2016-06-28'

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

def validate_with_map7():
    return 

def main():
    trn = pd.read_csv(train_path, low_memory=False)
    tst = pd.read_csv(test_path, low_memory=False)
    
    df = clean_data(trn, tst)
    
    trn, tst, features = data_engineering(df, use_dates, test_date)
    
    X_trn, Y_trn, X_vld, Y_vld = split_data(trn, features)
    
    dtrn = xgb.DMatrix(X_trn, label=Y_trn, feature_names=features)
    dvld = xgb.DMatrix(X_vld, label=Y_vld, feature_names=features)
    watch_list = [(dtrn, 'train'), (dvld, 'eval')]
    model = xgb.train(param, dtrn, num_boost_round=10, evals=watch_list, early_stopping_rounds=20)
    
    pickle.dump(model, open("./model/xgb.baseline.pkl", "wb"))
    best_ntree_limit = model.best_ntree_limit

    
    #dall = xgb.DMatrix(X_all, label=Y_all, feature_names=features)
    #best_ntree_limit = int(best_ntree_limit * (len(XY_trn) + len(XY_vld)) / len(XY_trn))
    
    print("Feature importance:")
    for kv in sorted([(k,v) for k,v in model.get_fscore().items()], key=lambda kv: kv[1], reverse=True):
        print(kv)
        
        
    X_tst = tst[features].values
    dtst = xgb.DMatrix(X_tst, feature_names=features)
    preds_tst = model.predict(dtst, ntree_limit=best_ntree_limit)
    ncodpers_tst = tst['ncodpers'].values
    #preds_tst = preds_tst - tst.as_matrix(columns=[prod + '_prev' for prod in prods])
    preds_tst = preds_tst - tst[[prod + '_prev' for prod in prods]].values
    
    submit_file = open('./model/xgb.baseline.2015-06-28', 'w')
    submit_file.write('ncodpers,added_products\n')
    for ncodper, pred in zip(ncodpers_tst, preds_tst):
        y_prods = [(y,p,ip) for y,p,ip in zip(pred, prods, range(len(prods)))]
        y_prods = sorted(y_prods, key=lambda a: a[0], reverse=True)[:7]
        y_prods = [p for y,p,ip in y_prods]
        submit_file.write('{},{}\n'.format(int(ncodper), ' '.join(y_prods)))

    
if __name__ == '__main__':
    main()
    