import pandas as pd


def encoding(x, test_data):
    # encoded the data
    encoded_training_predictors = pd.get_dummies(x)
    encoded_testing_predictors = pd.get_dummies(test_data)
    final_train, final_test = encoded_training_predictors.align(
        encoded_testing_predictors, join='left', axis=1)
    return final_train, final_test
