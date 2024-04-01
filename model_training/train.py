import yaml
import pandas as pd

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

    features_selected += config_dict['numeric_columns'] # Using numeric varibles as they are

    for feat_eng_type, temp_dict in config_dict['feature_engineering'].items():
        features_selected += temp_dict['engineered_features']

    return features_selected

    
if __name__ == '__main__':

    dataset_path = '../feature_engineering/train_labeled_small.csv'
    df_dataset = load_local_data(dataset_path)

    data_config_path = '../feature_engineering/feature_engineering.yaml'
    feature_eng_config_dict = read_lists_from_yaml(data_config_path)

    data_config_path = 'training_config.yaml'
    training_config_dict = read_lists_from_yaml(data_config_path)

    model_save_path = 'sample_model.joblib'



    x_features = feature_selection(feature_eng_config_dict)
    y_feature = 'y'

    X_trn, Y_trn, X_vld, Y_vld = take_data_pd(df_dataset,
                                              feature_eng_config_dict['key_timestamp'],
                                              training_config_dict['train_dates'],
                                              training_config_dict['val_dates'],
                                              x_features,
                                              y_feature
                                  )

    multiclass_clf = RandomForestClassifier(
        # params=training_config_dict['random_forest_parameters']
    )

    multiclass_clf = XGBoostClassifier(
        # params=training_config_dict['random_forest_parameters']
    )

    # Standardizing features
    multiclass_clf.set_scaler(X_trn)
    X_train_scaled = multiclass_clf.scale_data(X_trn)
    X_val_scaled = multiclass_clf.scale_data(X_vld)

    # Train and evaluate RandomForestClassifier
    multiclass_clf.fit(X_train_scaled, X_val_scaled, Y_trn, Y_vld)



    # Evaluation
    rf_pred = multiclass_clf.predict(X_val_scaled)
    print("Random Forest Classifier Report:")
    print(classification_report(Y_vld, rf_pred))

    multiclass_clf.export_model(model_save_path)

    pass