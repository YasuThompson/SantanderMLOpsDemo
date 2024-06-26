import yaml
import numpy as np
import pandas as pd

from db_connection import query_from_db

def load_local_data(csv_path):
    return pd.read_csv(csv_path, sep=',', low_memory=False)

def read_lists_from_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        return data

class FeatureEngineering:
    def __init__(self, if_keep_input_features=False):
        """

        :param if_keep_input_features:
        """
        self.input_features = None
        self.engineered_features = None
        self.param_dict = None
        self.if_keep_input_features = if_keep_input_features
    def apply_feat_eng(self, state):
        raise NotImplementedError("")

    def get_feat_eng_dict(self):

        if self.input_features is None or self.engineered_features is None:
            raise ValueError("Inputs and outputs of the feature engineering must be specified")
        else:
            feat_eng_dict = {
                'input_features': self.input_features,
                'engineered_features': self.engineered_features
            }

            if self.param_dict is not None:
                feat_eng_dict['parameters'] = self.param_dict

        return feat_eng_dict





class FeatEngLabelEncoding(FeatureEngineering):
    def __init__(self):
        super().__init__()


    def apply_feat_eng(self, df, input_features):
        self.input_features = input_features
        self.engineered_features = []
        for col in self.input_features:
            col_engineered = '{}_encoded'.format(col)
            df[col_engineered], _ = df[col].factorize()
            self.engineered_features.append(col_engineered)

        return df


class FeatEngDateEncoding(FeatureEngineering):
    def __init__(self):
        super().__init__()

    def safe_convert_to_month(self, date):
        try:
            return date.month
        except AttributeError:  # Catch AttributeError if 'date' is not in the correct format
            return 0  # Replace with NaN or any other suitable value

    def safe_convert_to_year(self, date):
        try:
            return date.year
        except AttributeError:  # Catch AttributeError if 'date' is not in the correct format
            return 0  # Replace with NaN or any other suitable value

    def apply_feat_eng(self, df, input_features):
        self.input_features = input_features
        self.engineered_features = []

        for col in input_features:
            month_col_engineered = '{}_month'.format(col)
            year_col_engineered = '{}_year'.format(col)
            df[month_col_engineered] = df[col].apply(self.safe_convert_to_month).astype(np.int8).fillna(-99)
            df[year_col_engineered] = df[col].apply(self.safe_convert_to_year).astype(np.int16).fillna(-99)
            self.engineered_features.append(year_col_engineered)
            self.engineered_features.append(month_col_engineered)

        return df


class FeatEngLag(FeatureEngineering):
    def __init__(self):
        super().__init__()

    def date_to_int(self, str_date, start_year=2015):
        Y, M, D = [int(a) for a in str_date.strip().split("-")]
        int_date = (int(Y) - start_year) * 12 + int(M)
        return int_date

    def apply_feat_eng(self, df, date_column, customer_id_column, lag_features, lag_size):

        self.input_features = [date_column] + [customer_id_column] + lag_features

        # First, converting timestamps to integers denoting months
        # TODO: This process might not be needed in the future. You could simply calculate monthly lags
        df['int_date'] = df[date_column].map(self.date_to_int).astype(np.int8)

        # Copy the data, add 1 to the integer date, and generate lag. Add '_prev' to the variable name.
        df_lag = df.copy()
        df_lag = df_lag[[customer_id_column, 'int_date'] + lag_features ]
        for col in lag_features:
            df_lag = df_lag.rename(columns={col: col + '_prev'})
        df_lag['int_date'] += 1

        # 原本データと lag データを ncodperと int_date を基準として合わせます。lag データの int_dateは 1 だけ押されているため、前の月の製品情報が挿入されます。
        df = df.merge(df_lag, on=[customer_id_column, 'int_date'], how='left')

        del df_lag

        # 前の月の製品情報が存在しない場合に備えて、0に代替します。
        for col in lag_features:
            # prev = col + '_prev'
            prev = '{}_prev'.format(col)
            df[prev] = df[prev].fillna(0)

        # TODO: we might need different type of filling based on lag columns
        # df = df.fillna(-99)

        self.engineered_features = ['{}_prev'.format(col) for col in lag_features]
        self.param_dict = {'lag_size': lag_size}

        return df

def data_engineering(df, config_dict):

    feat_eng_dict = {}

    # (Feature engineering) label encoding
    label_encoder = FeatEngLabelEncoding()
    df = label_encoder.apply_feat_eng(df, config_dict['categorical_columns'])
    feat_eng_dict['label_encoding'] = label_encoder.get_feat_eng_dict()

    # (Feature engineering) date encoding
    date_encoder = FeatEngDateEncoding()
    df = date_encoder.apply_feat_eng(df, config_dict['date_columns'])
    feat_eng_dict['date_encoding'] = date_encoder.get_feat_eng_dict()

    lag_calculator = FeatEngLag()
    date_column = config_dict['key_timestamp']
    customer_id_column = config_dict['customer_id']
    lag_features = config_dict['product_columns']
    lag_size = 1
    df = lag_calculator.apply_feat_eng(df,
                                       date_column,
                                       customer_id_column,
                                       lag_features,
                                       lag_size
                                       )

    feat_eng_dict['lag_1'] = lag_calculator.get_feat_eng_dict()


    return df, feat_eng_dict

def label_data(df, config_dict, label_column='y'):
    pass
    # Ex
    X = []
    Y = []

    for i, prod in enumerate(config_dict['product_columns']):
        prev = prod + '_prev'

        index_extracted = (df[prod] == 1) & (df[prev] == 0)
        # df[index_extracted]['y'] += ',{}'.format(i)

        prX = df[index_extracted]
        prY = np.zeros(prX.shape[0], dtype=np.int8) + i
        X.append(prX)
        Y.append(prY)
    XY = pd.concat(X)
    Y = np.hstack(Y)
    XY[label_column] = Y

    return XY



if __name__ == '__main__':
    data_source = 'workflow_test'

    train_path = '../data_prep/train_cleaned.csv'
    data_config_path = '../data_prep/data_prep_config_cleaned.yaml'

    config_dict = read_lists_from_yaml(data_config_path)

    save_path_feature_engineered = 'train_feature_engineered.csv'
    save_path_labeled = 'train_labeled.csv'
    config_save_path = 'feature_engineering.yaml'

    if data_source=='csv':
        print("Loading data from csv files")
        df_train = load_local_data(train_path)

        print("Feature engineering")
        df_feature_engineered, feat_eng_dict = data_engineering(df_train, config_dict)

        label_column = 'y'
        XY = label_data(df_feature_engineered, config_dict, label_column=label_column)
        config_dict['label_column'] = label_column

        df_feature_engineered.to_csv(save_path_feature_engineered, index=False)
        # df_labeled.to_csv(save_path_labeled, index=False)
        XY.to_csv(save_path_labeled, index=False)

        config_dict['feature_engineering'] = feat_eng_dict


        with open(config_save_path, 'w') as file:
            yaml.dump(config_dict, file)

    elif data_source == 'workflow_test':
        table_name = 'santander_cleaned_sample'
        query = f"SELECT * FROM {table_name}"
        columns, rows = query_from_db(query)