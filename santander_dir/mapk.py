import numpy as np

def apk(actual, predicted, k=7, default=0.0):
    # AP@7なので、最大7個まで使用します。
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        # 点数を付与する条件は次のとおり :
        # 予測値が正答に存在し (‘p in actual’)
        # 予測値に重複がなければ (‘p not in predicted[:i]’) 
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    # 正答値が空白である場合、ともかく 0.0点を返します。
    if not actual:
        return default

    # 正答の個数(len(actual))として average precisionを求めます。
    return score / min(len(actual), k)

def mapk(actual, predicted, k=7, default=0.0):
    # list of listである正答値(actual)と予測値(predicted)から顧客別 Average Precisionを求め, np.mean()を通して平均を計算します。
    return np.mean([apk(a, p, k, default) for a, p in zip(actual, predicted)]) 
