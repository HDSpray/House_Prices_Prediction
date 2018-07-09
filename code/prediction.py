import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

y = train_data.SalePrice
X = train_data.drop(['SalePrice'], axis=1)

# split the data set
train_X, test_X, train_y , test_y = train_test_split(X, y, random_state=17)

# resolve missing value
imputer = Imputer()
train_X = imputer.fit_transform(train_X)
test_X = imputer.transform(test_X)

# train the model
model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
model.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(test_X, test_y)], verbose=False) 

# test the accuary
val_predictions = model.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(val_predictions, test_y)))
