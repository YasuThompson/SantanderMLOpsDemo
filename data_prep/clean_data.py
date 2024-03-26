import yaml
import numpy as np
import pandas as pd

def take_value_dict(key, dict):
    try:
        return dict[key]
    except:
        return key

def load_local_data(csv_path):
    return pd.read_csv(csv_path, sep=',', low_memory=False)


def read_lists_from_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        return data


def clean_data_pandas(df):

    # Getting some column names and their mappings from a yaml file
    config_dict = read_lists_from_yaml('data_prep/data_prep_config.yaml')

    column_map_dict = config_dict['col_name_map_dict'] # A dictionary mapping column names
    target_cols = config_dict['target_cols'] # Target columns to predict

    # Renaming column names
    df = df.rename(columns=column_map_dict)

    # Renaming columns
    target_cols = [take_value_dict(col, column_map_dict) for col in target_cols]

    # When product information is nan, replacing them with 0
    df[target_cols] = df[target_cols].fillna(0.0).astype(np.int8)

    # Deliting customers who have no products (could harm prediction)
    no_product = df[target_cols].sum(axis=1) == 0
    df = df[~no_product]


    # Filling nan values of numeric data
    df[column_map_dict['age']] = df[column_map_dict['age']].replace(' NA', -99)
    df[column_map_dict['age']] = df[column_map_dict['age']].astype(np.int8)

    df[column_map_dict['antiguedad']] = df[column_map_dict['antiguedad']].replace('     NA', -99)
    df[column_map_dict['antiguedad']] = df[column_map_dict['antiguedad']].astype(np.int8)

    df[column_map_dict['renta']] = df[column_map_dict['renta']].replace('         NA', -99)
    df[column_map_dict['renta']] = df[column_map_dict['renta']].fillna(-99)
    df[column_map_dict['renta']] = df[column_map_dict['renta']].astype(float).astype(np.int8)

    df[column_map_dict['indrel_1mes']] = df[column_map_dict['indrel_1mes']].replace('P', 5)
    df[column_map_dict['indrel_1mes']] = df[column_map_dict['indrel_1mes']].fillna(-99)
    df[column_map_dict['indrel_1mes']] = df[column_map_dict['indrel_1mes']].astype(float).astype(np.int8)

    return df


if __name__ == '__main__':
    data_source = 'csv'
    train_path = 'data/santander-product-recommendation/train_ver2_small.csv'

    save_path = 'data_prep/train_cleaned_small.csv'

    if data_source=='csv':
        print("Loading data from csv files")
        df_train = load_local_data(train_path)
        print("Cleaninng data.")
        df_cleaned = clean_data_pandas(df_train)
        df_cleaned.to_csv(save_path, index=False)
