import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from missing_data import imputate_missing_data, drop_col_with_missing
from encoding import encoding
from model import xgboost_model, forest_model

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
y = train_data.SalePrice
X = train_data.drop(['SalePrice'], axis=1)
encoded_train, encoded_test = encoding(X, test_data)

# handle missing data
final_data = []
final_data.append(imputate_missing_data(encoded_train.copy(), encoded_test.copy()))
final_data.append(drop_col_with_missing(encoded_train.copy(), encoded_test.copy()))

# split the data set
for data in final_data:
    (data['train_X'], data['test_X'], data['train_y'], 
            data['test_y']) = train_test_split(data['final_train'], y, random_state=43)

# train the models
for data in final_data:
    model = (xgboost_model(data['train_X'], data['train_y'], data['test_X'], data['test_y']))
    val_predictions = model.predict(data['test_X'])
    print("Mean Absolute Error : " + str(mean_absolute_error(val_predictions, data['test_y'])))
    model = (forest_model(data['train_X'], data['train_y']))
    val_predictions = model.predict(data['test_X'])
    print("Mean Absolute Error : " + str(mean_absolute_error(val_predictions, data['test_y'])))

# output
# ouput_predictions = model.predict(output_test)
# print(ouput_predictions)


def submission(test_data, ouput_predictions):
    my_submission = pd.DataFrame({
            'Id' : test_data.Id, 
            'SalePrice' : ouput_predictions
        })
    my_submission.to_csv('submission.csv', index=False)

