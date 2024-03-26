import numpy as np

def load_local_data(csv_path):
    return pd.read_csv(csv_path, sep=',', low_memory=False)

def data_engineering(df, use_dates, test_dates):
    features = []

    features += categorical_cols
    features += ['age', 'antiguedad', 'renta', 'ind_nuevo', 'indrel', 'indrel_1mes', 'ind_actividad_cliente']

    # (特徴量エンジニアリング) 2つの日付変数から年度と月の情報を抽出します。
    df['fecha_alta_month'] = df['fecha_alta'].map(
        lambda x: 0.0 if x.__class__ is float else float(x.split('-')[1])).astype(np.int8)
    df['fecha_alta_year'] = df['fecha_alta'].map(
        lambda x: 0.0 if x.__class__ is float else float(x.split('-')[0])).astype(np.int16)
    features += ['fecha_alta_month', 'fecha_alta_year']

    df['ult_fec_cli_1t_month'] = df['ult_fec_cli_1t'].map(
        lambda x: 0.0 if x.__class__ is float else float(x.split('-')[1])).astype(np.int8)
    df['ult_fec_cli_1t_year'] = df['ult_fec_cli_1t'].map(
        lambda x: 0.0 if x.__class__ is float else float(x.split('-')[0])).astype(np.int16)
    features += ['ult_fec_cli_1t_month', 'ult_fec_cli_1t_year']

    # それ以外の変数の欠損値をすべて -99に代替します。
    df = df.fillna(-99)

    # 日付を数字に変換する関数です。 2015-01-28は 1, 2016-06-28は 18に変換します。
    def date_to_int(str_date):
        Y, M, D = [int(a) for a in str_date.strip().split("-")]
        int_date = (int(Y) - 2015) * 12 + int(M)
        return int_date

    # 日付を数字に変換し int_dateに保存します。
    df['int_date'] = df['fecha_dato'].map(date_to_int).astype(np.int8)

    # データをコピーし, int_date 日付に1を加え lagを生成します。変数名に _prevを追加します。
    df_lag = df.copy()
    df_lag.columns = [col + '_prev' if col not in ['ncodpers', 'int_date'] else col for col in df.columns]
    df_lag['int_date'] += 1

    # 原本データと lag データを ncodperと int_date を基準として合わせます。lag データの int_dateは 1 だけ押されているため、前の月の製品情報が挿入されます。
    df_trn = df.merge(df_lag, on=['ncodpers', 'int_date'], how='left')

    # メモリの効率化のために、不必要な変数をメモリから除去します。
    del df, df_lag

    # 前の月の製品情報が存在しない場合に備えて、0に代替します。
    for prod in prods:
        prev = prod + '_prev'
        df_trn[prev] = df_trn[prev].fillna(0)
    df_trn = df_trn.fillna(-99)

    # lag-1 変数を追加します。
    features += [feature + '_prev' for feature in features]
    features += [prod + '_prev' for prod in prods]

    ###
    ### Baseline モデル以後、多様な特徴量エンジニアリングをここに追加します。
    ###

    ## モデル学習
    # 学習のため、データを訓練、検証用に分離します。
    # 学習には 2016-01-28 ~ 2016-04-28 のデータだけを使用し、検証には 2016-05-28 のデータを使用します。

    trn = df_trn[df_trn['fecha_dato'].isin(use_dates)]
    tst = df_trn[df_trn['fecha_dato'].isin(test_dates)]
    del df_trn

    return trn, tst, features


if __name__ == '__main__':
    cleaned_data_path = 'data_cleaned.csv'
    df_cleaned = load_local_data(cleaned_data_path)

    trn, tst, features = data_engineering(df, use_dates, test_dates)