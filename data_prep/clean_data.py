import yaml
import numpy as np
import pandas as pd

def take_value_dict(key, dict):
    try:
        return dict[key]
    except:
        return key

def load_local_data(csv_path):
    df =  pd.read_csv(csv_path, sep=',', low_memory=False)
    df = df.drop('Unnamed: 0', axis=1)
    return df


def read_lists_from_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        return data

def clean_num_element(x, replace_val=-99):
    """
    Cleans an element in an object column of a pandas dataframe
    :param x:
    :param replace_val:
    :return:
    """

    if pd.isna(x) == True:
        replaced_value = replace_val

    else:
        try:
            replaced_value = int(x)
        except:
            try:
                replaced_value = float(x)
            except:
                replaced_value = replace_val

    return replaced_value





def clean_data_pandas(df, config_dict):

    # Getting some column names and their mappings from a yaml file
    column_map_dict = config_dict['col_name_map_dict'] # A dictionary mapping column names

    # Renaming column names
    df = df.rename(columns=column_map_dict)

    # Renaming columns
    config_dict_cleaned = {}
    config_dict_cleaned['numeric_columns'] = [take_value_dict(col, column_map_dict) for col in config_dict['numeric_columns']]
    config_dict_cleaned['categorical_columns'] = [take_value_dict(col, column_map_dict) for col in config_dict['categorical_columns']]
    config_dict_cleaned['date_columns'] = [take_value_dict(col, column_map_dict) for col in config_dict['date_columns']]
    config_dict_cleaned['product_columns'] = [take_value_dict(col, column_map_dict) for col in config_dict['product_columns']]

    # When product information is nan, replacing them with 0
    df[config_dict_cleaned['product_columns']] = df[config_dict_cleaned['product_columns']].fillna(0.0).astype(np.int8)

    # Deleting customers who have no products (could harm prediction)
    no_product = df[config_dict_cleaned['product_columns']].sum(axis=1) == 0
    df = df[~no_product]

    # Filling nan values of numeric data
    for num_col in config_dict_cleaned['numeric_columns']:
        df[num_col] = df[num_col].apply(lambda x: clean_num_element(x))

    # (Feature engineering) label encoding
    # Convert date columns to datetime objects
    for date_col in config_dict_cleaned['date_columns']:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    return df, config_dict_cleaned


if __name__ == '__main__':
    data_source = 'csv'
    train_path = 'data/santander-product-recommendation/train_ver2_small.csv'
    data_config_path = 'data_prep/data_prep_config.yaml'

    data_save_path = 'data_prep/train_cleaned_small.csv'
    config_save_path = 'data_prep/data_prep_config_cleaned.yaml'

    config_dict = read_lists_from_yaml(data_config_path)

    if data_source=='csv':
        print("Loading data from csv files")
        df_train = load_local_data(train_path)
        print("Cleaninng data.")
        df_cleaned, config_dict_cleaned = clean_data_pandas(df_train, config_dict)

        df_cleaned.to_csv(data_save_path, index=False)

        with open(config_save_path, 'w') as file:
            yaml.dump(config_dict_cleaned, file)
