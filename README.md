# Santander MLOps Demo 

![Alt text](images/santander_mlops_design.png?raw=true "Overview")
[More detailed overview can be found here.](https://miro.com/app/board/uXjVKaGxblU=/?share_link_id=865469751187) 

An MLOps demo using Santander Product Recommendation Kaggle Competition Dataset. I chose the dataset for the following reasons. 
- There are relatively a good amount of data (13647309 rows for training data, 929615 rows for test data)
- The task is relatively simple (multi class classification)
- The data are relatively messy and in Spanish, so it is close to real world scenarios.
- **Data science part is not really the point this time,** so let's not elaborate models and their trainings in the beginning. 

# Rule
- Please notify which task you work on to everyone
- Task with one indentation: ask Yasuto before starting 
- Task with two indentations: you don't need to ask me, but please share results
- Start with the smalles computations to save cloud cost (e.g.don't process all the data, don't do thorough grid search)
- Suggest any necessary tasks if you think it's needed, and make them into small pieces and write in the TODO list. 
- Be open to any questions on issues

# (Starting point) how to run locally 
## 1. Data preparation
Download the dataset from [Santander Product Recommendation Kaggle competition page](https://www.kaggle.com/c/santander-product-recommendation) and place them in `data` directory. And run the following
```
cd data_prep
python training/clean_data.py 
cd ..
```
## 2. Feature enginering
```
cd feature_engineering
python feature_engineering.py 
cd ..
```
## 3. Model training
```
cd model_training
python train.py 
cd ..
```
## 4. Inference
UNDER  CONSTRUCTION


# TODO
- To make a local workflow
  - [x] Data cleaning 
  - [ ] Feature engineering
    - [x] Finishing basic script
    - [x] Defining features in declarative ways
    - [ ] Adding more feature engineering
  - [ ] Model training
      - [x] Finishing the basic script
      - [ ] Cross validation
      - [ ] Implementing general ML class both for scikit-learn and XGBoost
      - [ ] Simple grid search
  - [ ] Inference
- EDA
  - [ ] EDA ノートの修正、翻訳
  - [ ] Connecting notebook with a database for ad hoc analysis. 
- Database
  - [x] Making a database project
  - [ ] Writing a script to sequentially uploading monthly records to the database (maybe with SQLAlchemy)
    - [x] SQLAlchemy (but it might not be effective this way)
    - [ ] Finding other solutions
  - [ ] Writing code to clean the raw data
    - [x] Pandas
    - [ ] SQL 
    - [ ] Pyspark
- Feature engineering
  - [ ] Writing the current feature engineering, which is currently in Pandas.
    - [x] Pandas
    - [ ] SQL 
    - [ ] Pyspark
  - [ ] Checking feature store options
    - [ ] MLflow
    - [ ] Kubeflow (Feast?)
    - [ ] Other general databases
- Model training
  - [ ] Making a general ML model/training class 
    - [x] Making scikit-learn model training script (the lighter the better)
    - [x] Making XGBoost model training script
    - [ ] Elaborating the model classes with more features (parameter tuning etc.)
  - [ ] Making evaluation script
    - [ ] Making evaluation script
    - [ ] Making cross-validation script
  - [ ] Writing parameter tuning code
- Model registry
  - [ ] Checking model registry options
    - [ ] MLflow 
    - [ ] Kubeflow
    - [ ] Other general storage
- Model serving
  - [ ] (Survey) how do you preprocess test data when3 you inference?
  - [ ] Checking model serving options
    - [ ] MLflow 
    - [ ] Kubeflow
    - [ ] Other general NoSQL
  - [ ] Making an endpoint with either of the options above
- Kaggle upload and monitoring
  - [ ] Finishing the kaggle upload container
  - [ ] Making a monitoring logic based on Kaggle upload
  - [ ] Raising an alert for concept drift or whatever errors
- Making Web App
  - [ ] Making a BI (web app) 
    - [ ] Making a local visualization with plotly 
    - [ ] Making local BI demo with Streamlit or Plotly Dash
    - [ ] Deploying the visualisation
    - [ ] Enable uploading files from local to the UI

* M1の学生のタスク
  * 実際のKaggleのコンペに参加していると想定して、それぞれのディレクトリ内での処理が正しいか確認して、パフォーマンスをあげてください。
  * それぞれのタスクを Issues か Slack で宣言して、一つのブランチを作ってその中で完成させてプルリク送ってください。
  * そして、最終的にはそれぞれの処理を Dockerfile と Docker-compose として記述してください。
  * タスクは上のTODOリストから選ぶか、より詳細なタスクを加えてください。
  * インフラ側は基本的に各データエンジニアリング、データサイエンス的に何が起きているのかには、あまり関与しないので。
  * ただ、自分と城居は相談には乗るよ（多分 Slack で）。データサイエンス部分は田村に気軽に相談してください。
  * そのうちこういうUIの作成も考えています (https://dash.gallery/Portal/)


# Members and roles
- Yasuto: 言い出しっぺ（２年前から）。マネージャー。基本何でもやる。現無職。
- Takumi: Fujitsu. Kubernetes要員。
- (Potentially) Rikuya: Yahoo.

皆さんの参加待っています。

### Reference：
* [Santander Product Recommendation](https://www.kaggle.com/c/santander-product-recommendation)
* [MLOps guideline by Databricks](https://www.databricks.com/resources/ebook/the-big-book-of-mlops)