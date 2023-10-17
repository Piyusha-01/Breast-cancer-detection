from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()
print(data.keys())
import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
p=df.head()
print(p);
df.info()
X = data.data
y = data.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.25, random_state = 0)
print("---------------------------------------------------------------------------------------")
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train);
print(X_test);
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(y_pred);
from sklearn.metrics import accuracy_score
act=accuracy_score( y_test, y_pred)
print(act);
acpt=accuracy_score(y_train, classifier.predict(X_train))
print(acpt);
