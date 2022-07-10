from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

mushrooms = pd.read_csv('mushrooms.csv', index_col=False, header=0)
mushroomsLabel = mushrooms.iloc[:,0:1]

le = LabelEncoder()
mushroomsLabel = le.fit_transform(mushroomsLabel)
ohe = OneHotEncoder()
mushroomsEncoded = ohe.fit_transform(mushrooms)
mushroomsEncoded = pd.DataFrame(mushroomsEncoded.toarray())
mushroomsNoLabel = mushroomsEncoded.iloc[:,1:119]

X_train, X_test, label_train, label_test = train_test_split(mushroomsNoLabel, mushroomsLabel,
                                                            test_size=0.999, random_state = 2)

#AdaBoost model
model_ada = AdaBoostClassifier()
model_ada.fit(X_train, label_train)

label_pred_ada = model_ada.predict(X_test)
predictions_ada = [round(value) for value in label_pred_ada]

accuracy_ada = accuracy_score(label_test, predictions_ada)
mean_accuracy_ada = np.mean(accuracy_ada)
std_accuracy_ada = np.std(accuracy_ada)
print ("Średnia dokładność AdaBoost: %.3f (+- %.2f)" % (mean_accuracy_ada, std_accuracy_ada))

print("Macierz pomyłek AdaBoost:")
confusion_matrix_ada = confusion_matrix(label_test, predictions_ada)
print(confusion_matrix_ada)

#XGBoost model
model_xgb = XGBClassifier()
model_xgb.fit(X_train, label_train)

label_pred_xgb = model_xgb.predict(X_test)
predictions_xgb = [round(value) for value in label_pred_xgb]

accuracy_xgb = accuracy_score(label_test, predictions_xgb)
mean_accuracy_xgb = np.mean(accuracy_xgb)
std_accuracy_xgb = np.std(accuracy_xgb)
print ("Średnia dokładność XGBoost: %.3f (+- %.2f)" % (mean_accuracy_xgb, std_accuracy_xgb))

print("Macierz pomyłek XGBoost:")
confusion_matrix_xgb = confusion_matrix(label_test, predictions_xgb)
print(confusion_matrix_xgb)