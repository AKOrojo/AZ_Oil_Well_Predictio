import joblib
import pandas as pd

y_test = pd.read_csv("C:/Users/DELL/OneDrive/Azubi Africa/data/processed/y_test.csv")
X_test_maxabs = pd.read_csv("C:/Users/DELL/OneDrive/Azubi Africa/data/processed/X_test_maxabs.csv")
X_test = pd.read_csv("C:/Users/DELL/OneDrive/Azubi Africa/data/processed/X_test.csv")
X_test = X_test.drop(X_test.columns[0], axis=1)
y_test = y_test.drop(X_test.columns[0], axis=1)

X_test_maxabs = X_test_maxabs.to_numpy()
X_test = X_test.to_numpy()

Ada = joblib.load("C:/Users/DELL/OneDrive/Azubi Africa/model/ada_best_joblib")
pred = Ada.predict(X_test)

New = pd.DataFrame(X_test)
New['actual'] = y_test
New['predicted'] = pred

New.to_csv("C:/Users/DELL/OneDrive/Azubi Africa/submissions/predictions_samples.csv")


