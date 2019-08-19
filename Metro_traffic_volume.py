import pandas as pd
import scipy as sp
import sklearn
import warnings
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression
warnings.simplefilter(action="ignore",category=FutureWarning)
df=pd.read_csv("Metro_Interstate_Traffic_Volume.csv")
print(df.shape)
print(df.columns)
print(df.weather_description.unique())
def process_categorical_features():
    dummies_weather_description=pd.get_dummies(df['weather_description'],prefix='weather',drop_first=True)
    dummies_date_time=pd.get_dummies(df['date_time'],prefix='date',drop_first=True)
    df.drop(['weather_description','date_time'],axis=1,inplace=True)
    return pd.concat([df,dummies_weather_description,dummies_date_time],axis=1)
df_=process_categorical_features()
X=df.drop(['traffic_volume','holiday','weather_main'],axis=1)
Y=df['traffic_volume']
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.33, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
lm = LinearRegression()
lm.fit(X_train, Y_train)
Y_pred = lm.predict(X_test)
plt.scatter(Y_test, Y_pred)
plt.xlabel("WEATHER AND TIME: $Y_i$")
plt.ylabel("PREDICTED TRAFFIC: $\hat{Y}_i$") 
plt.title("WEATHER AND TIME VS PREDICTED TRAFFIC: $Y_i$ vs $\hat{Y}_i$")
mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print(mse)
from sklearn.ensemble import GradientBoostingRegressor
clf = GradientBoostingRegressor()
clf.fit(X_train,Y_train)
predicted = clf.predict(X_test)
expected = Y_test
plt.figure(figsize=(4, 3))
plt.scatter(predicted, expected)
plt.plot([0, 50], [0, 50], '--k')
plt.axis('tight')
plt.xlabel('True traffic')
plt.ylabel('Predicted traffic')
plt.tight_layout()
mse = sklearn.metrics.mean_squared_error(predicted,expected)
print(mse)





