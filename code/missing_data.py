from sklearn.preprocessing import Imputer


def imputate_missing_data(train_X, test):
    cols_with_missing = (col for col in train_X.columns
                         if train_X[col].isnull().any() or test[col].isnull().any())
    for col in cols_with_missing:
        train_X[col + '_was_missing'] = train_X[col].isnull()
        test[col + '_was_missing'] = test[col].isnull()

    # Imputation
    imputer = Imputer()
    train_X = imputer.fit_transform(train_X)
    output_test = imputer.transform(test)
    return {'final_train': train_X, 'final_test': output_test}


def drop_col_with_missing(train_X, test):
    cols_with_missing = (col for col in train_X.columns
                         if train_X[col].isnull().any() or test[col].isnull().any())

    reduced_original_data = train_X.drop(cols_with_missing, axis=1)
    reduced_original_test = test.drop(cols_with_missing, axis=1)
    return {'final_train': reduced_original_data, 'final_test': reduced_original_test}
