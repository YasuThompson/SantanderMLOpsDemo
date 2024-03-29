import yaml
import numpy as np
import pandas as pd

def load_local_data(csv_path):
    return pd.read_csv(csv_path, sep=',', low_memory=False)

def read_lists_from_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        return data

def data_engineering(df, data_config_path='data_prep_config.yaml'):

    # Getting some column names and their mappings from a yaml file
    config_dict = read_lists_from_yaml(data_config_path)
    column_map_dict = config_dict['col_name_map_dict']  # A dictionary mapping column names
    target_cols = config_dict['target_cols']  # Target columns to predict
    target_cols = [column_map_dict[col] for col in target_cols]

    #
    # features = []
    # features += [column_map_dict[col] for col in target_cols]
    # features += ['age', 'antiguedad', 'renta', 'ind_nuevo', 'indrel', 'indrel_1mes', 'ind_actividad_cliente']
    #

    # (Feature engineering) label encoding
    for col in target_cols:
        df[col], _ = df[col].factorize()




    # (Feature engineering) taking year and month information from some timestamp columns
    # TODO: making the processes belwo more readable
    df[column_map_dict['fecha_alta']+'_month'] = df[column_map_dict['fecha_alta']].map(
        lambda x: 0.0 if x.__class__ is float else float(x.split('-')[1])).astype(np.int8)
    df[column_map_dict['fecha_alta']+'_year'] = df[column_map_dict['fecha_alta']].map(
        lambda x: 0.0 if x.__class__ is float else float(x.split('-')[0])).astype(np.int16)

    df[column_map_dict['ult_fec_cli_1t']+'_month'] = df[column_map_dict['ult_fec_cli_1t']].map(
        lambda x: 0.0 if x.__class__ is float else float(x.split('-')[1])).astype(np.int8)
    df[column_map_dict['ult_fec_cli_1t']+'_year'] = df[column_map_dict['ult_fec_cli_1t']].map(
        lambda x: 0.0 if x.__class__ is float else float(x.split('-')[0])).astype(np.int16)



    # (Feature engineering) label encoding
    # Convert date columns to datetime objects
    # df[column_map_dict['fecha_alta']] = pd.to_datetime(df[column_map_dict['fecha_alta']], errors='coerce').fillna(0.0)
    # df[column_map_dict['ult_fec_cli_1t']] = pd.to_datetime(df[column_map_dict['ult_fec_cli_1t']], errors='coerce').fillna(0.0)

    # # Extract year and month information
    # df[column_map_dict['fecha_alta_month']] = df[column_map_dict['fecha_alta']].dt.month.astype(np.int8)
    # df[column_map_dict['fecha_alta_year']] = df[column_map_dict['fecha_alta']].dt.year.astype(np.int16)
    #
    # df[column_map_dict['ult_fec_cli_1t_month']] = df[column_map_dict['ult_fec_cli_1t']].dt.month.astype(np.int8)
    # df[column_map_dict['ult_fec_cli_1t_year']] = df[column_map_dict['ult_fec_cli_1t']].dt.year.astype(np.int16)

    # Filling the rest nan values
    df = df.fillna(-99)

    # Converts dates to integers, starting from 2015-01-28.
    # e.g. 2015-01-28 to 1, 2016-06-28 to 18
    def date_to_int(str_date):
        Y, M, D = [int(a) for a in str_date.strip().split("-")]
        int_date = (int(Y) - 2015) * 12 + int(M)
        return int_date

    # 日付を数字に変換し int_dateに保存します。
    df['int_date'] = df[column_map_dict['fecha_dato']].map(date_to_int).astype(np.int8)

    # Copy the data, add 1 to the integer date, and generate lag. Add '_prev' to the variable name.
    df_lag = df.copy()
    df_lag.columns = [col + '_prev' if col not in [column_map_dict['ncodpers'], 'int_date'] else col
                      for col in df.columns]
    df_lag['int_date'] += 1

    # 原本データと lag データを ncodperと int_date を基準として合わせます。lag データの int_dateは 1 だけ押されているため、前の月の製品情報が挿入されます。
    df = df.merge(df_lag, on=[column_map_dict['ncodpers'], 'int_date'], how='left')

    # 前の月の製品情報が存在しない場合に備えて、0に代替します。
    for col  in target_cols:
        prev = col + '_prev'
        df[prev] = df[prev].fillna(0)
    df = df.fillna(-99)

    # # lag-1 変数を追加します。
    # features += [feature + '_prev' for feature in features]
    # features += [prod + '_prev' for prod in prods]

    # trn = df_trn[df_trn['fecha_dato'].isin(use_dates)]

    return df


if __name__ == '__main__':
    data_source = 'csv'
    train_path = 'data_prep/train_cleaned_small.csv'

    save_path = 'feature_engineering/train_feature_engineered_small.csv'

    if data_source=='csv':
        print("Loading data from csv files")
        df_train = load_local_data(train_path)
        print("Feature engineering")
        df_feature_engineered = data_engineering(df_train)
        df_feature_engineered.to_csv(save_path, index=False)