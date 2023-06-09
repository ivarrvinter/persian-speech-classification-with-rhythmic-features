import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#import joblib

df = pd.read_csv('intensity.csv')

X = df[['rPVIm', 'nPVIm', 'rPVIp', 'nPVIp']]
y = df['speaker']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=49)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(f_classif, k=4)),
    ('logreg', LogisticRegression(C=14.0, penalty='l2', solver='saga', max_iter=900))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

report = classification_report(y_test, y_pred, zero_division=1)
print(report)

#joblib.dump(pipeline, 'intensity-pipeline.joblib')
#joblib.load('intensity-pipeline.joblib')

