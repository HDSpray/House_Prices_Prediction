import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

y = train_data.SalePrice
X = train_data.drop(['SalePrice'], axis=1)

# endcoded the data
encoded_training_predictors = pd.get_dummies(X)

# split the data set
train_X, test_X, train_y , test_y = 
    train_test_split(encoded_training_predictors, y, random_state=17)

# Bachup the data and fix the missing value
imputed_X_train_plus = train_X.copy()
imputed_X_test_plus = test_X.copy()
cols_with_missing = (col for col in train_X.columns if train_X[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

# Imputation
imputer = Imputer()
train_X = imputer.fit_transform(imputed_X_train_plus)
test_X = imputer.transform(imputed_X_test_plus)

# train the model
model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
model.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(test_X, test_y)], verbose=False) 

# test the accuary
val_predictions = model.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(val_predictions, test_y)))
