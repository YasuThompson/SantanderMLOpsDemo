import yaml
import numpy as np
import pandas as pd

def load_local_data(csv_path):
    return pd.read_csv(csv_path, sep=',', low_memory=False)

def read_lists_from_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        return data


def data_engineering(df, config_dict):
    cat_cols = config_dict['categorical_columns']  # Target columns to predict
    prod_cols = config_dict['product_columns']

    # (Feature engineering) label encoding
    for col in cat_cols:
        df[col], _ = df[col].factorize()

    def safe_convert_to_month(date):
        try:
            return date.month
        except AttributeError:  # Catch AttributeError if 'date' is not in the correct format
            return 0  # Replace with NaN or any other suitable value

    def safe_convert_to_year(date):
        try:
            return date.year
        except AttributeError:  # Catch AttributeError if 'date' is not in the correct format
            return 0  # Replace with NaN or any other suitable value

    # (Feature engineering) label encoding
    # Convert date columns to datetime objects
    # Extract year and month information
    df['first_holder_date_month'] = df['first_holder_date'].apply(safe_convert_to_month).astype(np.int8)
    df['first_holder_date_year'] = df['first_holder_date'].apply(safe_convert_to_year).astype(np.int16)

    df['primary_last_date_month'] = df['primary_last_date'].apply(safe_convert_to_month).astype(np.int8)
    df['primary_last_date_year'] = df['primary_last_date'].apply(safe_convert_to_year).astype(np.int16)

    # Filling the rest nan values
    df = df.fillna(-99)

    # Converts dates to integers, starting from 2015-01-28.
    # e.g. 2015-01-28 to 1, 2016-06-28 to 18
    def date_to_int(str_date, start_year=2015):
        Y, M, D = [int(a) for a in str_date.strip().split("-")]
        int_date = (int(Y) - start_year) * 12 + int(M)
        return int_date

    # 日付を数字に変換し int_dateに保存します。
    df['int_date'] = df['date'].map(date_to_int).astype(np.int8)

    # Copy the data, add 1 to the integer date, and generate lag. Add '_prev' to the variable name.
    df_lag = df.copy()
    df_lag.columns = [col + '_prev' if col not in ['customer_code', 'int_date'] else col
                      for col in df.columns]
    df_lag['int_date'] += 1

    # 原本データと lag データを ncodperと int_date を基準として合わせます。lag データの int_dateは 1 だけ押されているため、前の月の製品情報が挿入されます。
    df = df.merge(df_lag, on=['customer_code', 'int_date'], how='left')

    # 前の月の製品情報が存在しない場合に備えて、0に代替します。
    for col in prod_cols:
        prev = col + '_prev'
        df[prev] = df[prev].fillna(0)
    df = df.fillna(-99)

    return df

def label_data(df, config_dict):
    pass
    # 訓練データから新規購買件数だけを抽出します。
    X = []
    Y = []

    df['y'] = np.nan

    for i, prod in enumerate(config_dict['product_columns']):
        prev = prod + '_prev'

        index_extracted = (df[prod] == 1) & (df[prev] == 0)
        df[index_extracted]['y'] = i

        prX = df[index_extracted]
        prY = np.zeros(prX.shape[0], dtype=np.int8) + i
        X.append(prX)
        Y.append(prY)
    XY = pd.concat(X)
    Y = np.hstack(Y)
    XY['y'] = Y

    return df, XY



if __name__ == '__main__':
    data_source = 'csv'
    train_path = 'data_prep/train_cleaned_small.csv'
    data_config_path = 'data_prep/data_prep_config_cleaned.yaml'

    config_dict = read_lists_from_yaml(data_config_path)

    save_path_feature_engineered = 'feature_engineering/train_feature_engineered_small.csv'
    save_path_labeled = 'feature_engineering/train_labeled_small.csv'
    save_path_labeled_debug = 'feature_engineering/train_labeled_small_debug.csv'

    if data_source=='csv':
        print("Loading data from csv files")
        df_train = load_local_data(train_path)

        print("Feature engineering")
        df_feature_engineered = data_engineering(df_train, config_dict)

        df_labeled, XY = label_data(df_feature_engineered, config_dict)

        df_feature_engineered.to_csv(save_path_feature_engineered, index=False)
        df_labeled.to_csv(save_path_labeled, index=False)
        XY.to_csv(save_path_labeled_debug, index=False)