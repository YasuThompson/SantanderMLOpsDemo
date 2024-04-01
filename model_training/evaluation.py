import yaml
import pandas as pd
import joblib

from multiclass_classifier import RandomForestClassifier, XGBoostClassifier
from sklearn.metrics import classification_report

def load_local_data(csv_path):
    return pd.read_csv(csv_path, sep=',', low_memory=False)


def read_lists_from_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        return data


def take_data_pd(df, key_timestamp_column, trn_dates, val_dates, x_features, y_feature):
    # 訓練、検証データに分離します。
    XY_trn = df[df[key_timestamp_column].isin(trn_dates)]
    XY_vld = df[df[key_timestamp_column].isin(val_dates)]

    # 訓練、検証データを XGBoost 形態に変換します。
    X_trn = XY_trn[x_features].values
    Y_trn = XY_trn[y_feature].values
    # dtrn = xgb.DMatrix(X_trn, label=Y_trn, feature_names=features)

    X_vld = XY_vld[x_features].values
    Y_vld = XY_vld[y_feature].values
    # dvld = xgb.DMatrix(X_vld, label=Y_vld, feature_names=features)

    return X_trn, Y_trn, X_vld, Y_vld


def feature_selection(config_dict):
    """
    Selects and returns features to use from engineered features
    Currently just selects all the features and engineered features in the yaml file
    # TODO: making a logic for selecting variables with high importance
    :return:
    """

    features_selected = []

    features_selected += config_dict['numeric_columns']  # Using numeric varibles as they are

    for feat_eng_type, temp_dict in config_dict['feature_engineering'].items():
        features_selected += temp_dict['engineered_features']

    return features_selected


if __name__ == '__main__':
    dataset_path = '../feature_engineering/train_labeled.csv'
    df_dataset = load_local_data(dataset_path)

    data_config_path = '../feature_engineering/feature_engineering.yaml'
    feature_eng_config_dict = read_lists_from_yaml(data_config_path)

    data_config_path = 'training_config.yaml'
    training_config_dict = read_lists_from_yaml(data_config_path)

    model_path = 'sample_model.joblib'

    x_features = feature_selection(feature_eng_config_dict)
    y_feature = 'y'

    X_trn, Y_trn, X_vld, Y_vld = take_data_pd(df_dataset,
                                              feature_eng_config_dict['key_timestamp'],
                                              training_config_dict['train_dates'],
                                              training_config_dict['val_dates'],
                                              x_features,
                                              y_feature
                                              )

    _, _, customer_id_vld, _ = take_data_pd(df_dataset,
                                              feature_eng_config_dict['key_timestamp'],
                                              training_config_dict['train_dates'],
                                              training_config_dict['val_dates'],
                                              feature_eng_config_dict['customer_id'],
                                              y_feature
                                              )

    for prod in feature_eng_config_dict['product_columns']:
        prev = prod + '_prev'
        padd = prod + '_add'
        X_vld[padd] = X_vld[prod] - X_vld[prev]

    multiclass_clf = joblib.load(model_path)

    # Evaluation
    rf_pred = multiclass_clf.predict(X_vld)

    feature_eng_config_dict['customer_id']
    # 検証データの予測上位7個を抽出します。
    result_vld = []
    for ncodper, pred in zip(ncodpers_vld, preds_vld):
        y_prods = [(y, p, ip) for y, p, ip in zip(pred, prods, range(len(prods)))]
        y_prods = sorted(y_prods, key=lambda a: a[0], reverse=True)[:7]
        result_vld.append([ip for y, p, ip in y_prods])

