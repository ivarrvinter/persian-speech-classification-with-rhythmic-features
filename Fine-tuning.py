import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import numpy as np

df = pd.read_csv('intensity.csv')

X = df[['rPVIm', 'nPVIm', 'rPVIp', 'nPVIp']]
y = df['speaker']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=49)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    'C': np.linspace(0.001, 20.0, 40),
    'penalty': ['l1', 'l2'],
    'solver': ['saga']
}

model = LogisticRegression(random_state=49, max_iter=10000)

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print("Best Parameters:", best_params)

y_pred = best_model.predict(X_test_scaled)

report = classification_report(y_test, y_pred, zero_division=1)
print(report)