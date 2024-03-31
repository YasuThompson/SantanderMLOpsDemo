from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
class BaseMultiClassClassifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

class XGBoostClassifier(BaseMultiClassClassifier):
    def __init__(self, xgb_params=None):
        if xgb_params is None:
            self.xgb_params = {}  # You can set default parameters here
        else:
            self.xgb_params = xgb_params

    def fit(self, x_trn, y_trn, x_vld, y_vld):
        import xgboost as xgb

        dtrn = xgb.DMatrix(x_trn, label=y_trn)
        dvld = xgb.DMatrix(x_vld, label=y_vld)
        watch_list = [(dtrn, 'train'), (dvld, 'eval')]
        self.model = xgb.train(self.param,
                               dtrn,
                               num_boost_round=10,
                               evals=watch_list,
                               early_stopping_rounds=1)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class RandomForestClassifier(BaseMultiClassClassifier):
    def __init__(self, rf_params=None):
        if rf_params is None:
            self.rf_params = {}  # You can set default parameters here
        else:
            self.rf_params = rf_params

    def fit(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(**self.rf_params)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)



class SVMClassifier(BaseMultiClassClassifier):
    def __init__(self, svm_params=None):
        super().__init__()
        if svm_params is None:
            self.svm_params = {}  # You can set default parameters here
        else:
            self.svm_params = svm_params

    def fit(self, X, y):
        from sklearn.svm import SVC
        self.model = SVC(**self.svm_params)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

if __name__ == '__main__':
    # Load Iris dataset
    # iris = load_iris()
    # X, y = iris.data, iris.target

    # Load Wine dataset
    wine = load_wine()
    X, y = wine.data, wine.target

    # # Load Breast Cancer dataset
    # cancer = load_breast_cancer()
    # X, y = cancer.data, cancer.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # # Train and evaluate XGBoostClassifier
    # xgb_clf = XGBoostClassifier()
    # xgb_clf.fit(X_train_scaled, y_train)
    # xgb_pred = xgb_clf.predict(X_test_scaled)
    # print("XGBoost Classifier Report:")
    # print(classification_report(y_test, xgb_pred))

    # Train and evaluate RandomForestClassifier
    rf_clf = RandomForestClassifier()
    rf_clf.fit(X_train_scaled, y_train)
    rf_pred = rf_clf.predict(X_test_scaled)
    print("Random Forest Classifier Report:")
    print(classification_report(y_test, rf_pred))

    # svm_clf = SVMClassifier()
    # svm_clf.fit(X_train, y_train)
    # svm_pred = svm_clf.predict(X_test_scaled)
    # print("Support Vector Machine Report:")
    # print(classification_report(y_test, svm_pred))