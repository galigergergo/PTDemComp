import numpy as np
from data_utils import *
import optuna
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from neuralprophet import NeuralProphet, set_random_seed
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor


def get_or_create_experiment(experiment_name):
  if experiment := mlflow.get_experiment_by_name(experiment_name):
      return experiment.experiment_id
  else:
      return mlflow.create_experiment(experiment_name)


def evaluation_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "MSE": np.nan,
            "MAPE": np.nan,
            "sMAPE": np.nan,
            "R2": np.nan
        }
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "MSE": mean_squared_error(y_true, y_pred),
        "MAPE": np.mean(np.abs((y_true - y_pred) / (y_true+1))) * 100,
        "sMAPE": np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred) + 1e-8)/2)) * 100,
        "R2": r2_score(y_true, y_pred)
    }


def evaluation_metrics_df(df1, df2):
    # Create mask for rows without any NaN in either y_true or y_pred
    df1 = df1.set_index('ds')
    df2 = df2.set_index('ds')
    
    common_index = df1.index.intersection(df2.index)
    
    df1_filtered = df1.loc[common_index]
    df2_filtered = df2.loc[common_index]

    # Filter out rows with NaNs
    y_true_clean = df1_filtered.values
    y_pred_clean = df2_filtered.values

    return evaluation_metrics(y_true_clean, y_pred_clean)


###############################
######## NEURALPROPHET ########
###############################

def get_NP_fixed_params():
    return dict(epochs=10,
                drop_missing=True,
                n_lags=0,
                n_forecasts=7,
                daily_seasonality=False,
                weekly_seasonality=False)


def build_NP_model(params):
    set_random_seed(42)
    
    m = NeuralProphet(**params)
    
    m = m.add_lagged_regressor(["y_hist"], n_lags=4)

    # custom daily for weekend
    m.add_seasonality(name="daily_weekend", period=1, fourier_order=3, condition_name="is_weekend")
    m.add_seasonality(name="daily_weekday", period=1, fourier_order=3, condition_name="is_weekday")
    
    # custom weekly for educational holidays
    m.add_seasonality(name="weekly_education", period=7, fourier_order=3, condition_name="is_education")
    m.add_seasonality(name="weekly_edu_holiday", period=7, fourier_order=3, condition_name="is_edu_holiday")

    return m


def run_training_NP(df_train, df_val, filt_val, params, log_mlflow=False):
    m = build_NP_model(params)
                 
    metrics_train = m.fit(drop_future_and_lags(df_train), validation_df=drop_future_and_lags(df_val),
                          early_stopping=True,
                          progress=None)

    forecast = m.predict(drop_future_and_lags(df_val), raw=True)

    # Evaluate
    y_pred = forecast[['ds', 'step0', 'step1', 'step2', 'step3', 'step4', 'step5', 'step6']]
    y_val = df_val[['ds', 'y', 'y_1', 'y_2', 'y_3', 'y_4', 'y_5', 'y_6']]
    y_val = y_val[filt_val]
    metrics = evaluation_metrics_df(y_val, y_pred)

    if log_mlflow:
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        mlflow.log_params(params)

    forecast = m.predict(drop_future_and_lags(df_val), raw=False)
    
    return m, forecast, metrics


def get_objective_NP(df_train, df_val, filt_val):
    def objective(trial):
        with mlflow.start_run(nested=True):
            params = dict(newer_samples_weight=trial.suggest_int('newer_samples_weight', 1, 100),
                          newer_samples_start=trial.suggest_float('newer_samples_start', 0.001, 0.999),
                          yearly_seasonality=trial.suggest_int('yearly_seasonality', 1, 10),
                          seasonality_reg=trial.suggest_float('seasonality_reg', 0, 2),
                          seasonality_mode=trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']))
            params.update(get_NP_fixed_params())
        
            _, _, metrics = run_training_NP(df_train, df_val, filt_val, params, log_mlflow=True)
            
            return metrics['RMSE']

    return objective


def run_optuna_tuning_NP(df, database_url, experiment_name, n_trials=30, n_jobs=1):
    storage = optuna.storages.RDBStorage(url=database_url)
    experiment_id = get_or_create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    df_train, df_val, filt_val, _, _ = preprocess_data_NP(df)

    objective = get_objective_NP(df_train, df_val, filt_val)    
    
    with mlflow.start_run(experiment_id=experiment_id, nested=True):
        study = optuna.create_study(study_name=experiment_name, direction='minimize', storage=storage, load_if_exists=True)
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

        params = study.best_params
        params.update(get_NP_fixed_params())
        mlflow.log_params(params)
        mlflow.log_metric('best_rmse', study.best_value)
        mlflow.log_metric('best_mse', study.best_value ** 2)


def get_best_NP_model(df, study_name, database_url):
    storage = optuna.storages.RDBStorage(url=database_url)
    
    study = optuna.create_study(study_name=study_name, direction="minimize", storage=storage, load_if_exists=True)

    df_train, _, _, df_test, filt_test = preprocess_data_NP(df)
    
    params = study.best_params
    params.update(get_NP_fixed_params())

    m, forecast, metrics = run_training_NP(df_train, df_test, filt_test, params)

    return m, forecast, metrics


###############################
########### XGBoost ###########
###############################

def get_XG_fixed_params():
    return {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': 'gbtree'
    }


def run_training_XG(X_train, y_train, X_val, y_val, params, log_mlflow=False):
    # One model for all output steps
    base_model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
    model = MultiOutputRegressor(base_model)
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    
    metrics = evaluation_metrics(y_val, y_pred)

    if log_mlflow:
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        mlflow.log_params(params)
    
    return model, y_pred, metrics


def get_objective_XG(X_train, y_train, X_val, y_val):
    def objective(trial):
        with mlflow.start_run(nested=True):
            params = {
                'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
                'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            }
            params.update(get_XG_fixed_params())
        
            _, _, metrics = run_training_XG(X_train, y_train, X_val, y_val, params, log_mlflow=True)
            
            return metrics['RMSE']

    return objective


def run_optuna_tuning_XG(df, database_url, experiment_name, n_trials=30, n_jobs=1):
    storage = optuna.storages.RDBStorage(url=database_url)
    experiment_id = get_or_create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    X_train, y_train, X_val, y_val, _, _, _ = preprocess_data_XG(df)

    objective = get_objective_XG(X_train, y_train, X_val, y_val)    
    
    with mlflow.start_run(experiment_id=experiment_id, nested=True):
        study = optuna.create_study(study_name=experiment_name, direction='minimize', storage=storage, load_if_exists=True)
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

        params = study.best_params
        params.update(get_XG_fixed_params())
        mlflow.log_params(params)
        mlflow.log_metric('best_rmse', study.best_value)
        mlflow.log_metric('best_mse', study.best_value ** 2)


def get_best_XG_model(df, study_name, database_url):
    storage = optuna.storages.RDBStorage(url=database_url)
    
    study = optuna.create_study(study_name=study_name, direction="minimize", storage=storage, load_if_exists=True)

    X_train, y_train, _, _, X_test, y_test, feature_names = preprocess_data_XG(df)
    
    params = study.best_params
    params.update(get_XG_fixed_params())

    model, y_pred, metrics = run_training_XG(X_train, y_train, X_test, y_test, params)
    
    return model, y_pred, metrics, X_test, y_test, feature_names
