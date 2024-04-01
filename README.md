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
## 4. Inference


# TODO
- To make a local workflow
  - [x] Data cleaning 
  - [ ] Feature engineering
    - [x] Finishing basic script
    - [x] Defining features in declarative ways
  - [ ] Model training
      - [ ] Finishing the basic script
      - [ ] Cross validation
      - [ ] Implementing general ML class both for scikit-learn and XGBoost
      - [ ] Simple grid search
  - [ ] Inference
- EDA
  - [ ] EDA ノートの修正、翻訳
  - [ ] Connecting notebook with a database for ad hoc analysis. 
- Database
  - [ ] Making a database project
  - [ ] Writing a script to sequentially uploading monthly records to the database (maybe with SQLAlchemy)
  - [ ] Writing code to clean the raw data
    - [ ] SQL 
    - [ ] Pyspark
- Feature engineering
  - [ ] Writing the current feature engineering, which is currently in Pandas.
    - [ ] SQL 
    - [ ] Pyspark
  - [ ] Checking feature store options
    - [ ] MLflow
    - [ ] Kubeflow (Feast?)
    - [ ] Other general databases
- Model training
  - [ ] Making a general ML model/training class 
    - [ ] Making scikit-learn model training script (the lighter the better)
    - [ ] Making XGBoost model training script
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


# Members and roles
- Yasuto: 言い出しっぺ（２年前から）。マネージャー。基本何でもやる。現無職。
- Takumi: Fujitsu. Kubernetes要員。
- (Potentially) Rikuya: Yahoo.

皆さんの参加待っています。

### Reference：
* [Santander Product Recommendation](https://www.kaggle.com/c/santander-product-recommendation)
* [MLOps guideline by Databricks](https://www.databricks.com/resources/ebook/the-big-book-of-mlops)