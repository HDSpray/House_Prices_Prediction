import pandas as pd

def encoding(X, test_data):
    # endcoded the data
    encoded_training_predictors = pd.get_dummies(X)
    encoded_testing_predictors = pd.get_dummies(test_data)
    final_train, final_test = encoded_training_predictors.align(
        encoded_testing_predictors, join='left', axis=1)
    return (final_train, final_test)
