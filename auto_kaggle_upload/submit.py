from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate() 
csv_file_path = './data/portseguro/sample_submission.csv'
message = 'test_submit'
competition_id = 'porto-seguro-safe-driver-prediction'
api.competition_submit(csv_file_path, message, competition_id)

# 参考：https://www.currypurin.com/entry/kaggle-api-submit
