from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from mapk import mapk


class BaseMultiClassClassifier(ABC):
    # TODO: Don't use pandas dataframe
    def __init__(self, features):
        self.features = features
        self.weights = None
        self.scaler = None

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

class XGBoostClassifier(BaseMultiClassClassifier):
    def __init__(self, xgb_params=None):
        if xgb_params is None:
            self.xgb_params = {}  # You can set default parameters here
        else:
            self.xgb_params = xgb_params

    def fit(self, x_trn, x_vld, y_trn, y_vld):
        self.model = xgb.XGBClassifier(early_stopping_rounds=1)

        self.model.fit(x_trn,
                       y_trn,
                       eval_set=[(x_vld, y_vld)],
                       )

    def predict(self, X):
        return self.model.predict(X)

class RandomForestClassifier(BaseMultiClassClassifier):
    def __init__(self, rf_params=None):
        if rf_params is None:
            self.rf_params = {}  # You can set default parameters here
        else:
            self.rf_params = rf_params

    def fit(self, x_trn, x_vld, y_trn, y_vld):
        # TODO: to implement early stopping logic here

        self.model = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1)
        self.model.fit(x_trn, y_trn)

    def predict(self, X):
        return self.model.predict(X)



if __name__ == '__main__':
    # Load Iris dataset
    # iris = load_iris()
    # X, y = iris.data, iris.target

    # Load Wine dataset
    wine = load_wine()
    X, y = wine.data, wine.target

    # Load Breast Cancer dataset
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target

    # Split the data into training and testing sets
    X_train, X_val , y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # multiclass_clf = XGBoostClassifier()
    # # Standardizing features
    # multiclass_clf.set_scaler(X_train)
    # X_train_scaled = multiclass_clf.scale_data(X_train)
    # X_val_scaled = multiclass_clf.scale_data(X_val)
    #
    # # Train and evaluate XGBoostClassifier
    # multiclass_clf.fit(X_train_scaled, X_val_scaled, y_train, y_val)
    # xgb_pred = multiclass_clf.predict(X_val_scaled)
    # print("XGBoost Classifier Report:")
    # print(classification_report(y_val, xgb_pred))

    multiclass_clf = RandomForestClassifier()
    # Standardizing features
    multiclass_clf.set_scaler(X_train)
    X_train_scaled = multiclass_clf.scale_data(X_train)
    X_val_scaled = multiclass_clf.scale_data(X_val)

    # Train and evaluate RandomForestClassifier
    multiclass_clf.fit(X_train_scaled, X_val_scaled, y_train, y_val)
    rf_pred = multiclass_clf.predict(X_val_scaled)
    print("Random Forest Classifier Report:")
    print(classification_report(y_val, rf_pred))

    pass