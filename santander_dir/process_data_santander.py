import numpy as np 
import pandas as pd
import xgboost as xgb


prods = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
             'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
             'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
             'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

categorical_cols = ['ind_empleado', 'pais_residencia', 'sexo', 'tiprel_1mes', 'indresi', 'indext', 'conyuemp', 'canal_entrada', 'indfall', 'tipodom', 'nomprov', 'segmento']



def clean_data(trn, tst):
    
    # 製品の変数を別途に保存しておきます。
    # prods = trn.columns[24:].tolist()

    # 製品変数の欠損値をあらかじめ0に代替しておきます。
    trn[prods] = trn[prods].fillna(0.0).astype(np.int8)

    # 24個の製品を1つも保有していない顧客のデータを除去します。
    no_product = trn[prods].sum(axis=1) == 0
    trn = trn[~no_product]

    # 訓練データとテストデータを統合します。テストデータにない製品変数は0で埋めます。
    for col in trn.columns[24:]:
        tst[col] = 0
    df = pd.concat([trn, tst], axis=0)

    # 学習に使用する変数を入れるlistです。
    #features = []

    # カテゴリ変数を .factorize() 関数に通して label encodingします。

    for col in categorical_cols:
        df[col], _ = df[col].factorize()

    # df = pd.get_dummies(df, columns=categorical_cols)

    #features += categorical_cols

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
    #features += ['age','antiguedad','renta','ind_nuevo','indrel','indrel_1mes','ind_actividad_cliente']

    
    
    return df


def data_engineering(df, use_dates, test_dates):
    
    features = []
    
    features += categorical_cols
    features += ['age','antiguedad','renta','ind_nuevo','indrel','indrel_1mes','ind_actividad_cliente']
    
    # (特徴量エンジニアリング) 2つの日付変数から年度と月の情報を抽出します。
    df['fecha_alta_month'] = df['fecha_alta'].map(lambda x: 0.0 if x.__class__ is float else float(x.split('-')[1])).astype(np.int8)
    df['fecha_alta_year'] = df['fecha_alta'].map(lambda x: 0.0 if x.__class__ is float else float(x.split('-')[0])).astype(np.int16)
    features += ['fecha_alta_month', 'fecha_alta_year']

    df['ult_fec_cli_1t_month'] = df['ult_fec_cli_1t'].map(lambda x: 0.0 if x.__class__ is float else float(x.split('-')[1])).astype(np.int8)
    df['ult_fec_cli_1t_year'] = df['ult_fec_cli_1t'].map(lambda x: 0.0 if x.__class__ is float else float(x.split('-')[0])).astype(np.int16)
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
    df_lag.columns = [col + '_prev' if col not in ['ncodpers', 'int_date'] else col for col in df.columns ]
    df_lag['int_date'] += 1

    # 原本データと lag データを ncodperと int_date を基準として合わせます。lag データの int_dateは 1 だけ押されているため、前の月の製品情報が挿入されます。
    df_trn = df.merge(df_lag, on=['ncodpers','int_date'], how='left')

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


def split_data(trn, features, val_dates):
    
    # 訓練データから新規購買件数だけを抽出します。
    X = []
    Y = []
    for i, prod in enumerate(prods):
        prev = prod + '_prev'
        prX = trn[(trn[prod] == 1) & (trn[prev] == 0)]
        prY = np.zeros(prX.shape[0], dtype=np.int8) + i
        X.append(prX)
        Y.append(prY)
    XY = pd.concat(X)
    Y = np.hstack(Y)
    XY['y'] = Y

    # 訓練、検証データに分離します。

    #XY_trn = XY[XY['fecha_dato'] != vld_date]
    #XY_vld = XY[XY['fecha_dato'] == vld_date]
    XY_trn = XY[~XY['fecha_dato'].isin(val_dates)]
    XY_vld = XY[XY['fecha_dato'].isin(val_dates)]

    # 訓練、検証データを XGBoost 形態に変換します。
    X_trn = XY_trn[features].values
    Y_trn = XY_trn['y'].values
    #dtrn = xgb.DMatrix(X_trn, label=Y_trn, feature_names=features)

    X_vld = XY_vld[features].values
    Y_vld = XY_vld['y'].values
    #dvld = xgb.DMatrix(X_vld, label=Y_vld, feature_names=features)
    
    return X_trn, Y_trn, X_vld, Y_vld



    
    
    
    