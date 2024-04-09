import os
import sys
import joblib
import yaml
import pandas as pd
import xgboost as xgb

sys.path.append(os.path.join(os.path.dirname(__file__), '../model_training'))
import multiclass_classifier
from mapk import mapk
def load_local_data(csv_path):
    return pd.read_csv(csv_path, sep=',', low_memory=False)

def read_lists_from_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        return data



def feature_selection(config_dict):
    """
    Selects and returns features to use from engineered features
    Currently just selects all the features and engineered features in the yaml file
    # TODO: making a logic for selecting variables with high importance
    :return:
    """

    features_selected = []

    features_selected += config_dict['numeric_columns'] # Using numeric varibles as they are

    for feat_eng_type, temp_dict in config_dict['feature_engineering'].items():
        features_selected += temp_dict['engineered_features']

    return features_selected

def take_data_pd(df, key_timestamp_column, trn_dates, val_dates, x_features, y_feature):
    # 訓練、検証データに分離します。
    XY_trn = df[df[key_timestamp_column].isin(trn_dates)]
    XY_vld = df[df[key_timestamp_column].isin(val_dates)]

    # # 訓練、検証データを XGBoost 形態に変換します。
    # X_trn = XY_trn[x_features]
    # Y_trn = XY_trn[y_feature]
    # # dtrn = xgb.DMatrix(X_trn, label=Y_trn, feature_names=features)
    #
    # X_vld = XY_vld[x_features]
    # Y_vld = XY_vld[y_feature]
    # # dvld = xgb.DMatrix(X_vld, label=Y_vld, feature_names=features)

    return XY_trn, XY_vld


if __name__ == '__main__':

    dataset_path = '../feature_engineering/train_labeled.csv'
    df_dataset = load_local_data(dataset_path)

    data_config_path = '../feature_engineering/feature_engineering.yaml'
    feature_eng_config_dict = read_lists_from_yaml(data_config_path)

    data_config_path = '../model_training/training_config.yaml'
    training_config_dict = read_lists_from_yaml(data_config_path)

    trained_model_path = '../sample_model.joblib'
    loaded_model = joblib.load(trained_model_path)

    x_features = feature_selection(feature_eng_config_dict)
    y_feature = 'y'

    XY_trn, XY_vld = take_data_pd(df_dataset,
                                              feature_eng_config_dict['key_timestamp'],
                                              training_config_dict['train_dates'],
                                              training_config_dict['val_dates'],
                                              x_features,
                                              y_feature
                                              )

    # dvld = xgb.DMatrix(X_vld, label=Y_vld, feature_names=x_features)
    preds_vld = loaded_model.model.predict_proba(XY_vld[x_features])

    ncodpers_vld = XY_vld[feature_eng_config_dict['customer_id']].values

    product_list = feature_eng_config_dict['product_columns']

    # 検証データから新規購買を求めます。
    for prod in feature_eng_config_dict['product_columns']:
        prev = prod + '_prev'
        padd = prod + '_add'
        XY_vld[padd] = XY_vld[prod] - XY_vld[prev]
    add_vld = XY_vld[[prod + '_add' for prod in product_list]].values

    add_vld_dict = {ncodper: list() for ncodper in ncodpers_vld}
    for ncodper_idx, ncodper in enumerate(ncodpers_vld):
        for prod_idx, prod in enumerate(product_list):
            if add_vld[ncodper_idx, prod_idx] > 0:
                add_vld_dict[ncodper].append(prod_idx)

    result_vld = []
    for ncodper, pred in zip(ncodpers_vld, preds_vld):
        y_prods = [(y, p, ip) for y, p, ip in zip(pred, product_list, range(len(product_list)))]
        # print(y_prods)
        y_prods = sorted(y_prods, key=lambda a: a[0], reverse=True)[:7]
        # print(y_prods)
        result_vld.append([ip for y, p, ip in y_prods])

    print(mapk(list(add_vld_dict.values()), result_vld, 7, 0.0))
