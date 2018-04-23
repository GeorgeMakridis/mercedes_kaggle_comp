import pandas as pd
from sklearn.preprocessing import RobustScaler
from config import *



def read_data():
    train = pd.read_csv(PATH+TRAIN_CSV)
    test = pd.read_csv(PATH+TEST_CSV)
    return train, test


def prepare_and_scale_data(train, test):
    y_train = train['y'].values
    id_test = test['ID']
    num_train = len(train)
    df_all = pd.concat([train, test])
    df_all.drop(['ID', 'y'], axis=1, inplace=True)
    """ One-hot encoding of categorical/strings """
    df_all = pd.get_dummies(df_all, drop_first=True)
    """Sscaling features"""
    scaler = RobustScaler()
    df_all = scaler.fit_transform(df_all)
    train = df_all[:num_train]
    test = df_all[num_train:]
    return train, y_train, test, id_test
