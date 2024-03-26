import numpy as np
import pandas as pd

def load_local_data(csv_path):
    return pd.read_csv(csv_path, sep=',', low_memory=False)

def clean_data(trn, tst, prod_cols, categorical_cols):
    # 製品変数の欠損値をあらかじめ0に代替しておきます。
    trn[prod_cols] = trn[prod_cols].fillna(0.0).astype(np.int8)

    # 24個の製品を1つも保有していない顧客のデータを除去します。
    no_product = trn[prod_cols].sum(axis=1) == 0
    trn = trn[~no_product]

    # 訓練データとテストデータを統合します。テストデータにない製品変数は0で埋めます。
    for col in trn.columns[24:]:
        tst[col] = 0
    df = pd.concat([trn, tst], axis=0)

    # カテゴリ変数を .factorize() 関数に通して label encodingします。
    for col in categorical_cols:
        df[col], _ = df[col].factorize()


    # 数値型変数の特異値と欠損値を -99に代替し、整数型に変換します。
    df['age'] = df['age'].replace(' NA', -99)
    df['age'] = df['age'].astype(np.int8)

    df['antiguedad'] = df['antiguedad'].replace('     NA', -99)
    df['antiguedad'] = df['antiguedad'].astype(np.int8)

    df['renta'] = df['renta'].replace('         NA', -99)
    df['renta'] = df['renta'].fillna(-99)
    df['renta'] = df['renta'].astype(float).astype(np.int8)

    df['indrel_1mes'] = df['indrel_1mes'].replace('P', 5)
    df['indrel_1mes'] = df['indrel_1mes'].fillna(-99)
    df['indrel_1mes'] = df['indrel_1mes'].astype(float).astype(np.int8)

    # 学習に使用する数値型変数を featuresに追加します。
    # features += ['age','antiguedad','renta','ind_nuevo','indrel','indrel_1mes','ind_actividad_cliente']

    return df


if __name__ == '__main__':

    data_source = 'csv'
    train_path = '../data/santander-product-recommendation/train_ver2_small.csv'
    test_path = '../data/santander-product-recommendation/test_ver2.csv'

    save_path = 'data_cleaned.csv'

    if data_source=='csv':
        print("Loading data from csv files")

        trn = load_local_data(train_path)
        tst = load_local_data(test_path)

    #### Phase 2 ####
    # DATA CLEANING.
    # Preferably completed before SQL server.
    print("Cleaninng data.")
    df_cleaned = clean_data(trn, tst)


    df_cleaned.to_csv(save_path, index=False)
