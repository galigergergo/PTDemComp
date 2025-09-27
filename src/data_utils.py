import os
import pandas as pd

SHARED_PROJECTS_PATH = '...'


def get_dataframe(dataset):
    if dataset == 'M4':
        return pd.read_pickle(SHARED_PROJECTS_PATH)

def add_holidays_cond(df):
    df["is_edu_holiday"] = df["is_school_holiday"] | df["is_university_holiday"]
    df["is_education"] = (df["is_edu_holiday"]*-1 + 1)
    df = df.drop(columns=['is_public_holiday', 'is_school_holiday', 'is_university_holiday',
                          'is_school', 'is_university'])
    return df


def add_weekend_cond(df):
    df[df.ds.dt.dayofweek.isin([5, 6])]
    df["is_weekend"] = False
    df.loc[df.ds.dt.dayofweek.isin([5, 6]), "is_weekend"] = True
    df["is_weekday"] = False
    df.loc[~df.ds.dt.dayofweek.isin([5, 6]), "is_weekday"] = True
    return df


def drop_time_features(df):
    return df.drop(['hour', 'dayofweek', 'dayofyear', 'contyear'], axis=1)


def drop_future(df):
    return df.drop(['y_1', 'y_2', 'y_3', 'y_4', 'y_5', 'y_6'], axis=1)


def drop_lags(df):
    return df.drop(['y_1h', 'y_2h', 'y_3h', 'y_4h'], axis=1)


def drop_future_and_lags(df):
    return drop_lags(drop_future(df))


def preprocess_data_NP(df):
    df.loc[:, "y_hist"] = df["y"]

    df_train = df[df["train_np"]].copy()
    df_val = df.copy()
    df_test = df.copy()
    
    df_train = df_train.drop(["train_np", "train_xg", "val", "test"], axis=1)
    df_val = df_val.drop(["train_np", "train_xg", "val", "test"], axis=1)
    df_test = df_test.drop(["train_np", "train_xg", "val", "test"], axis=1)

    df_train = drop_time_features(df_train)
    df_val = drop_time_features(df_val)
    df_test = drop_time_features(df_test)
    df_train = add_holidays_cond(df_train)
    df_val = add_holidays_cond(df_val)
    df_test = add_holidays_cond(df_test)
    df_train = add_weekend_cond(df_train)
    df_val = add_weekend_cond(df_val)
    df_test = add_weekend_cond(df_test)

    filt_val = df['val']
    filt_test = df['test']
    
    return df_train, df_val, filt_val, df_test, filt_test


def preprocess_data_XG(df):
    df_train = df[df["train_xg"]].copy()
    df_val = df[df["val"]].copy()
    df_test = df[df["test"]].copy()
    
    df_train = df_train.drop(["train_np", "train_xg", "val", "test"], axis=1)
    df_val = df_val.drop(["train_np", "train_xg", "val", "test"], axis=1)
    df_test = df_test.drop(["train_np", "train_xg", "val", "test"], axis=1)

    df_train = add_holidays_cond(df_train)
    df_val = add_holidays_cond(df_val)
    df_test = add_holidays_cond(df_test)
    df_train = add_weekend_cond(df_train)
    df_val = add_weekend_cond(df_val)
    df_test = add_weekend_cond(df_test)

    X_train, y_train = df_train.drop(["ds", "y", "y_1", "y_2", "y_3", "y_4", "y_5", "y_6"], axis=1).values,\
                           df_train[["y", "y_1", "y_2", "y_3", "y_4", "y_5", "y_6"]].values
    X_val, y_val     = df_val.drop(["ds", "y", "y_1", "y_2", "y_3", "y_4", "y_5", "y_6"], axis=1).values,\
                           df_val[["y", "y_1", "y_2", "y_3", "y_4", "y_5", "y_6"]].values
    X_test, y_test   = df_test.drop(["ds", "y", "y_1", "y_2", "y_3", "y_4", "y_5", "y_6"], axis=1).values,\
                           df_test[["y", "y_1", "y_2", "y_3", "y_4", "y_5", "y_6"]].values

    feature_names = df_train.drop(["ds",'y',"y_1","y_2","y_3","y_4","y_5",'y_6'], axis=1).columns
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names
