import numpy as np;
from sklearn.linear_model import LogisticRegression;
from sklearn.metrics import accuracy_score;
import pandas as pd;
import plotly.express as px;

data = pd.read_csv('data.csv');
velocity = data['Velocity'].tolist();
escape = data['Escaped'].tolist();

X = np.reshape(velocity, (-1, 1));
Y = np.reshape(escape, (-1, 1));

lr = LogisticRegression();
model = lr.fit(X, Y.ravel());

Y_pred = lr.predict(X);

accuracy = accuracy_score(escape, Y_pred);
print(accuracy);