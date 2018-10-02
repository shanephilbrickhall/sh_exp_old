import numpy as np
import pandas as pd
from numpy import loadtxt, genfromtxt
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn import model_selection, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.metrics import mean_squared_error
import datetime
from datetime import datetime, timedelta
from collections import OrderedDict


def generate_month_list(start_date=None,end_date=None):
    # Function expects start date and end date in the format: YYYY-MM-DD
    dates= [start_date, end_date]
    start, end = [datetime.strptime(date, "%Y-%m-%d") for date in dates]
    return OrderedDict(((start + timedelta(date)).strftime(r"%b-%y"), None)
                       for date in range((end - start).days)).keys()


def pd_get_month_list(start_date=None, end_date=None):
    # Function expects start date and end date in the format: YYYY-MM-DD
    date1= start_date
    date2= end_date
    month_list = [i.strftime("%b-%y") for i in pd.date_range(start=date1, end=date2, freq='MS')]
    return month_list


def load_data(file_name= None, delim_type= None, header_rows=None,data_type=None,converter=None,use_cols=None):

    if delim_type == 'csv':
        delim = ','
    elif delim_type == 'tab':
        delim = '\t'
    elif delim_type == 'colon':
        delim = ':'
    elif delim_type == 'pipe':
        delim = '|'
    elif delim_type == 'space':
        delim = ' '
    else:
        return "You must include a delimiter type; csv, tab, colon, pipe, space"
    if not header_rows:
        header_skip = 0
    else:
        header_skip = int(header_rows)
    if converter:
        new_array = genfromtxt(str(file_name), delimiter=delim, skip_header=header_skip,
                               dtype=data_type, autostrip=True, converters=converter, usecols=use_cols)
    else:
        new_array = genfromtxt(str(file_name), delimiter=delim, skip_header=header_skip,
                               dtype=data_type, autostrip=True, usecols=use_cols)
    return new_array


def split_data_support_results(ndarray=None):
    # Assumes results are in right most column
    new_array = ndarray
    num_rows = new_array.shape[0]
    num_cols = new_array.shape[1]

    # Split data from results
    support_data = new_array[:,0:num_cols-1]
    results = new_array[:,num_cols-1]
    return support_data, results


def standard_model_run(data=None,train_percentage=None):
    print("Input data set matrix shape, ", data.shape)
    support_set, result_set = split_data_support_results(data)

    if train_percentage:
        train_break = train_percentage
    else:
        train_break = 0.33

    seed = 12

    X_train, X_test, y_train, y_test = train_test_split(support_set, result_set,
                                                        test_size=train_break, random_state=seed)
    model = XGBRegressor()
    model.fit(X_train, y_train)
    model_accuracy = model.score(X_test, y_test)
    return model, model_accuracy


def grid_search_opt(data=None,search_parameters=None,train_percentage=None,):
    # Function expects a ndarray data input
    # Pull in data set break it into testing and training data
    print("Input data set matrix shape", data.shape)
    support_set, result_set = split_data_support_results(data)

    if train_percentage:
        train_break = train_percentage
    else:
        train_break = 0.20

    seed = 12

    X_train, X_test, y_train, y_test = train_test_split(support_set, result_set,
                                                        test_size=train_break, random_state=seed)

    print("Resulting X_train and y_train shapes after input data split: ", X_train.shape, y_train.shape)
    # Initialize model for grid search testing
    model_opt_seed = XGBRegressor()

    if search_parameters:
        grid_search_model_opt = GridSearchCV(model_opt_seed, param_grid=search_parameters, verbose=1)
    else:
        parameters = {
            'learning_rate': [0.001, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0],
            'max_depth': [3, 4, 6],
            'min_child_weight': [1, 2, 4, 6],
            'silent': [1],
            'subsample': [0.2, 0.5, 0.8, 1.0],
            'colsample_bytree': [0.5, 0.8, 1.0],
            'n_estimators': [100, 500, 1000],
            'seed': [12]
        }
        grid_search_model_opt = GridSearchCV(model_opt_seed, param_grid=parameters, verbose=1)

    grid_search_model_opt.fit(X_train, y_train)
    # Returns the model object, the test accuracy,
    #  the grid search best run parameters, and the grid search best run accuracy
    return grid_search_model_opt,grid_search_model_opt.score(X_test,y_test), \
           grid_search_model_opt.best_params_, grid_search_model_opt.best_score_



converter_function = lambda x: float(x.decode('UTF-8').strip("$"))

precip_data = load_data(file_name='MA-025-pcp-all-8-2008-2018.csv', delim_type='csv',
                        header_rows=5, data_type=float, use_cols=1)
avg_tmp_data = load_data(file_name='MA-025-tavg-all-8-2008-2018.csv', delim_type='csv',
                         header_rows=5, data_type=float,use_cols=1)
max_tmp_data = load_data(file_name='MA-025-tmax-all-8-2008-2018.csv', delim_type='csv',
                         header_rows=5, data_type=float, use_cols=1)
min_tmp_data = load_data(file_name='MA-025-tmin-all-8-2008-2018.csv', delim_type='csv',
                         header_rows=5, data_type=float,use_cols=1)
iso_day_ahead = load_data(file_name='ISO-NE-MonthlyAvg.DayAhead.csv', delim_type='csv',
                          header_rows=1, data_type=float, use_cols=1, converter={1: converter_function})
iso_real_time = load_data(file_name='ISO-NE-MonthlyAvg.RealTime.csv',delim_type='csv',
                          header_rows=1, data_type=float, use_cols=1, converter={1: converter_function})

print('ISONE DAY AHEAD: ', iso_day_ahead.shape)
print('ISONE RT : ', iso_real_time.shape)
print('AVG TMP: ', avg_tmp_data.shape)
print('MAX TMP: ', max_tmp_data.shape)
print('MIN TMP: ', min_tmp_data.shape)
print('PRECIP: ', precip_data.shape)

# Remove headers from ISO CSV
iso_day_ahead = iso_day_ahead[0:iso_day_ahead.shape[0]-1]
iso_real_time = iso_real_time[0:iso_real_time.shape[0]-1]
print(iso_day_ahead.shape,iso_real_time.shape)

fin_data_set = np.stack((avg_tmp_data,max_tmp_data,min_tmp_data,precip_data,iso_real_time,iso_day_ahead),axis=-1)
print("Stacked input data shape ", fin_data_set.shape," rows 0 through 5: ",fin_data_set[0:5,:])

support_set, result_set = split_data_support_results(fin_data_set)

print("Here is the support and result data set shapes: ", support_set.shape, result_set.shape)

seed = 12
test_train_split = 0.20

X_train, X_test, y_train, y_test = train_test_split(support_set, result_set,
                                                    test_size=test_train_split, random_state=seed)
print("Here is the X_train and y_train shapes: ", X_train.shape, y_train.shape)

print("Here is the start of the non-grid-search optimized model")
model_01 = XGBRegressor()
model_01.fit(X_train, y_train)

print("Here is the model summary: ", model_01)

# Accuracy score
print("Pre-Grid Search Model Accuracy: %.2f%%" % (model_01.score(X_test,y_test)))

print("Here is the start of the grid-search optimized model")

model_02_opt_seed = XGBRegressor()

parameters = {
    'learning_rate': [0.001, 0.1, 0.2, 0.3, 0.8, 1.0],
    'max_depth': [3, 4, 6],
    'min_child_weight': [1, 6],
    'silent': [1],
    'subsample': [0.5, 0.8, 1.0],
    'colsample_bytree': [0.5, 0.8, 1.0],
    'n_estimators': [100, 500, 1000],
    'seed': [5]
}

grid_search_model_opt = GridSearchCV(model_02_opt_seed, param_grid=parameters, verbose=1)

grid_search_model_opt.fit(X_train,y_train)

print("Here are the results of the model grid search: ", grid_search_model_opt.best_params_,
      " the best score is",
      grid_search_model_opt.best_score_, )


opt_model_fin = XGBRegressor(**grid_search_model_opt.best_params_)
opt_model_fin.fit(X_train,y_train)

print("Post-Grid Search Model Accuracy: %.2f%%" % (opt_model_fin.score(X_test,y_test)))
