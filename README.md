# Santander MLOps Demo 

![Alt text](images/santander_mlops_design.png?raw=true "Course schedule")


An MLOps demo using Santander Product Recommendation Kaggle Competition Dataset. I chose the dataset for the following reasong. 
- There are relatively a good amount of data (13647309 rows for training data, 929615 rows for test data)
- The task is relatively simple (multi class classification)
- The data are relatively messy and in Spanish, so it is close to real world scenarios.

# TODO
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
    - [ ] Making XGboost model trainign script
  - [ ] Making evlauation script
    - [ ] Making evaluation script
    - [ ] Making cross-validation script
  - [ ] Writing parameter tuning code
- Model registry
  - [ ] Checking model registry optiosn
    - [ ] MLflow 
    - [ ] Kubeflow
    - [ ] Other general storage
- Model serving
  - [ ] Checking model serving optiosn
    - [ ] MLflow 
    - [ ] Kubeflow
    - [ ] Other general NoSQL
  - [ ] Making an endpoind with either of the options above
- Kaggle upload and monitoring
  - [ ] Finishign the kaggle upload container
  - [ ] Making a monitoring logic based on Kaggle upload
  - [ ] Raising an alert for concept drift or whatever errors
- Making Web App
  - [ ] Making a BI (web app) 
    - [ ] Making a local visualization with plotly 
    - [ ] Making local BI demo with Streamlit or Plotly Dash
    - [ ] Deploying the visualisation
    - [ ] Enable uploading files from local to the UI


### Reference：
* [Santander Product Recommendation](https://www.kaggle.com/c/santander-product-recommendation)
