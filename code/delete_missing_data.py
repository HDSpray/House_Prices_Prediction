import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

y = train_data.SalePrice
X = train_data.drop(['SalePrice'], axis=1)

# encoded the data
encoded_training_predictors = pd.get_dummies(X)
encoded_testing_predictors = pd.get_dummies(test_data)
final_train, final_test = encoded_training_predictors.align(
        encoded_testing_predictors, join='left', axis=1)

# split the data set
train_X, test_X, train_y , test_y = train_test_split(
        final_train, y, random_state=17)

# Backup the data and fix the missing value
imputed_X_train_plus = train_X.copy()
imputed_X_test_plus = test_X.copy()
imputed_final_test = final_test.copy()

cols_with_missing = (col for col in train_X.columns if train_X[col].isnull().any())
cols_with_missing_at_ouput = (col for col in final_test.columns if final_test[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()
    imputed_final_test[col + '_was_missing'] = imputed_final_test[col].isnull()

for col in cols_with_missing_at_ouput:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()
    imputed_final_test[col + '_was_missing'] = imputed_final_test[col].isnull()

# Imputation
imputer = Imputer()
train_X = imputer.fit_transform(imputed_X_train_plus)
test_X = imputer.transform(imputed_X_test_plus)
output_test = imputer.transform(imputed_final_test)

# train the model
model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
model.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(test_X, test_y)], verbose=False) 

# test the actuary
val_predictions = model.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(val_predictions, test_y)))

# output
output_predictions = model.predict(output_test)
print(output_predictions)

my_submission = pd.DataFrame({
        'Id': test_data.Id,
        'SalePrice': output_predictions
    })
my_submission.to_csv('submission.csv', index=False)

