import pandas as pd
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

train_y = train_data.SalePrice
train_X = train_data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
