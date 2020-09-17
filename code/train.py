from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
import joblib
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

new_data = pd.read_csv("C:/Users/DELL/OneDrive/Azubi Africa/data/processed/processed_data_stage_1.csv").dropna()

# Separate features and labels
features = new_data[new_data.columns[0:14]]
label = new_data[new_data.columns[14]]
X, y = features.values, label.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale Data
max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X_train)
X_test_maxabs = max_abs_scaler.transform(X_test)

kfold = StratifiedKFold(n_splits=10)

# Adaboost
DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion": ["gini", "entropy"], "base_estimator__splitter": ["best", "random"],
                  "algorithm": ["SAMME", "SAMME.R"], "n_estimators": [1, 2],
                  "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1.5]}

gsadaDTC = GridSearchCV(adaDTC, param_grid=ada_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

gsadaDTC.fit(X_train_maxabs, y_train)

ada_best = gsadaDTC.best_estimator_

joblib.dump(ada_best, "C:/Users/DELL/OneDrive/Azubi Africa/model/ada_best_joblib")


