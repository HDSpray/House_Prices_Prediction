from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

# xgboost model regressor model
def xgboost_model(train_X, train_y, test_X, test_y):
    model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
    model.fit(train_X, train_y, early_stopping_rounds=5, 
            eval_set=[(test_X, test_y)], verbose=False) 
    return model

# random forest regressor model
def forest_model(train_X, train_y):
    model = RandomForestRegressor()
    model.fit(train_X, train_y)
    return model
    
