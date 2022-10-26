import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")
dataset = pd.read_csv("work1.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y = y.astype('int')
X = X.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
inputt=[int(x) for x in "20 4 1 1".split(' ')]
final=[np.array(inputt)]
# print(final)
b = regressor.predict(final)
pickle.dump(regressor,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))