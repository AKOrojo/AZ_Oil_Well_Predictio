# Importing Models
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

pd.DataFrame(X_train).to_csv("C:/Users/DELL/OneDrive/Azubi Africa/data/processed/X_train.csv")
pd.DataFrame(X_test).to_csv("C:/Users/DELL/OneDrive/Azubi Africa/data/processed/X_test.csv")
pd.DataFrame(y_train).to_csv("C:/Users/DELL/OneDrive/Azubi Africa/data/processed/y_train.csv")
pd.DataFrame(y_test).to_csv("C:/Users/DELL/OneDrive/Azubi Africa/data/processed/y_test.csv")
pd.DataFrame(y_train).to_csv("C:/Users/DELL/OneDrive/Azubi Africa/data/processed/y_train.csv")
pd.DataFrame(X_test_maxabs).to_csv("C:/Users/DELL/OneDrive/Azubi Africa/data/processed/X_test_maxabs.csv")
pd.DataFrame(X_train_maxabs).to_csv("C:/Users/DELL/OneDrive/Azubi Africa/data/processed/X_train_maxabs.csv")