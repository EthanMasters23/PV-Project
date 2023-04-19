#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### Imports ###

import os
import pandas as pd
import numpy as np
import pvlib
import math
import re

import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score, explained_variance_score, mean_squared_log_error

from sklearn.preprocessing import MinMaxScaler
from pykalman import KalmanFilter
from sklearn.impute import KNNImputer

# Kalman Smoothing using R objects
import rpy2.robjects as robjects
# import R packages
from rpy2.robjects.packages import importr
imputeTS = importr('imputeTS') 
kalman_StructTs = robjects.r['na.kalman']
kalman_auto_arima = robjects.r['na.kalman']
sea_decom = robjects.r['na.seadec']
sea_split = robjects.r['na.seasplit']

# Loading all possible data paths

def execute(file):
    path_list = []
    cwd = os.getcwd()
    datapath = cwd + '/PVdata/'
    for dir in os.scandir(datapath):
        if '.' in dir.name:
            continue
        year_path = datapath + f"{dir.name}"
        for dir in os.scandir(year_path):
            if dir.name == file:
                month_datapath = year_path + f"/{dir.name}/"
                for dir in os.scandir(month_datapath):
                    if ".DS" in dir.name:
                        continue
                    path_list += [month_datapath + f"{dir.name}"]
    return path_list

# Loading specified data paths

def path_func(year,month,year_2,month_2,path_list,file):
    path_found = []
    for path in path_list:
        path_copy = path.lower()
        data = re.search(r"/(\d{4})/[a-z]*/([a-z]*)\.csv",path_copy).group(1,2)
        if year:
            if not path_found:
                if re.search(fr"{year}",data[0]) and re.search(fr"{month}",data[1]):
                    path_found += [path]
            else:
                if re.search(fr"{year}",data[0]) and re.search(fr"{month}",data[1]):
                    path_found += [path]
                    path_found = [path_found[1],path_found[0]]

        if year_2:
            if re.search(fr"{year_2}",data[0]) and re.search(fr"{month_2}",data[1]):
                path_found += [path]

    print(f"\n Imputation methods for {file} and error metrics in {year}, {month} modeled with NaN values in {year_2}, {month_2}.\n")
    return path_found

# ======================== Data Pre-Preprocessing ============================ #

# Reshaping dataframe with timestamp index and feature

def reshape_df(df):
    df['DayID'] = df['DayID'].astype(str)
    df['TimeID'] = df['TimeID'].astype(str)
    df['date'] = df['DayID'] + 'T' +  df['TimeID']
    df = df.drop(columns = ['DayID','TimeID'])
    df.date = pd.to_datetime(df.date)
    df = df.set_index('date')
    df.index = df.index.tz_localize(tz = 'Etc/UTC')
    df = df.sort_index()
    return df

# Creating rows with NaN values where time gaps are larger than 21 seconds between observations

def add_missing_times(df):
    
    # creating of list of times to find interval gaps
    time_list = list(df.index)
    
    # calculating interval gaps if > 21s and storing [interval length (s), start_time, end_time]
    missing_intervals = [[(time_list[time+1] - time_list[time]).total_seconds(),time_list[time],time_list[time+1]]
                 for time in range(len(time_list)-1) if (time_list[time+1] - time_list[time]).total_seconds() > 21]
    # generating time stamps to fill interval gaps 
    interval_list = [element for sublist in [pd.date_range(start=interval[1],
                             end=interval[2]-pd.Timedelta(1,'s'),
                             freq='11s') for interval in missing_intervals] for element in sublist]
    
    full_intervals = [[(time_list[time+1] - time_list[time]).total_seconds(),time_list[time],time_list[time+1]]
                 for time in range(len(time_list)-1)]
    
    # checking for missing values at the beginning of the month
    if time_list[0] > time_list[0].replace(day=1,hour=1):
        print("Month found with missing values at the beginning of the month.")
        print('Time:',time_list[0])
        interval_list += [time for time in pd.date_range(start=time_list[0].replace(day=1,hour=0,minute=0,second=0),
                             end=time_list[0]-pd.Timedelta(1,'s'),
                             freq='11s')]
        missing_intervals += [[(time_list[0] - time_list[0].replace(day=1,hour=0,minute=0,second=0)).total_seconds(),
                             time_list[0].replace(day=1,hour=0,minute=0,second=0),time_list[0]]]
        
    # checking for missing values at the end of the month    
    next_month = time_list[0].replace(day=28,hour=0,minute=0,second=0) + pd.Timedelta(4,'d')
    last_day = next_month - pd.Timedelta(next_month.day,'d')
    if time_list[-1] < last_day.replace(hour = 23,minute=0):
        print("Month found with missing values at the end of the month.")
        print('Time:',time_list[-1])
        interval_list += [time for time in pd.date_range(start=time_list[-1],
                     end=last_day.replace(hour=23,minute=59,second=59),
                     freq='11s')]
        missing_intervals += [[(last_day.replace(hour=23,minute=59,second=59) - time_list[-1]).total_seconds(),
                             time_list[-1],last_day.replace(hour=23,minute=59,second=59)]]
        
    interval_list = list(set(interval_list))
    mt_df = pd.DataFrame(index=interval_list,columns=df.columns)
    mt_df.loc[interval_list] = np.nan
    df = pd.concat([df,mt_df], axis = 0).sort_index()

    return df,missing_intervals,full_intervals


# Removing night time observations, and irregular variable values

def remove_night(df):
    lat = 49.102
    lon = 6.215
    alt = 220
    solpos = pvlib.solarposition.get_solarposition(
        time=df.index,latitude=lat,longitude=lon,altitude=alt,method='pyephem')
    df = df[solpos['zenith'] <=90].replace(0,np.nan)
    return df

def remove_bad_temps(df):
    df['Temperature'] = np.where((df['Temperature'] > 60)|(df['Temperature'] < 0), np.nan, df['Temperature'])
    return df

def remove_bad_wind_speeds(df):
    df['WindSpeed'] = np.where((df['WindSpeed'] < 0)|(df['WindSpeed'] > 100), np.nan, df['WindSpeed'])
    return df

def remove_bad_dni(df):
    df['DirectIR'] = np.where((df['DirectIR'] > 2000)|(df['DirectIR'] < 0), np.nan, df['DirectIR'])
    return df

def remove_bad_dhi(df):
    df['DiffuseIR'] = np.where((df['DiffuseIR'] > 2000)|(df['DiffuseIR'] < 0), np.nan, df['DiffuseIR'])
    return df

# Dataframe cleaner function, takes in file path and loads preprocessed data

def df_cleaner(path_list,file):
    df_clean = pd.DataFrame()

    missing_intervals = []
    all_intervals = []

    for path in path_list:

        df_load = pd.read_csv(path,sep="\t|,",engine='python')
        if df_load.empty:
            continue
        if "Temprature" in df_load.columns:
            df_load.columns = ['DayID', 'TimeID', 'GlobalIR', 'DiffuseIR', 'DirectIR', 'WindSpeed', 'Temperature']

        # ==== reshaping df for timestap & adjusted headers ==== #
        df_load = reshape_df(df_load)

        # === filling gaps in time intervals === #
        df_load,intervals,full_intervals = add_missing_times(df_load)

        # # ==== Using PvLib to remove nightime values === #
        df_load = remove_night(df_load)
        
        if file == 'Irradiance':

            # # === Removing misread Temps === #
            df_load = remove_bad_temps(df_load)

            # # === Removing misread Wind Speeds === #
            df_load = remove_bad_wind_speeds(df_load)

            # # === Removing misread dni === #
            df_load = remove_bad_dni(df_load)

            # # === Removing misread dhi === #
            df_load = remove_bad_dhi(df_load)

        df_clean = pd.concat([df_clean,df_load],axis=0,ignore_index=False).sort_index()
        missing_intervals += intervals
        all_intervals += full_intervals
        
    return df_clean


# Additional preprocessing for performance evaluation script

def prep(pre_df, df_2):
    
    pre_df = pre_df.dropna(axis=0)
    if 'GlobalIR' in pre_df.columns:
        pre_df,df_2 = pre_df.drop(['GlobalIR'],axis=1),df_2.drop(['GlobalIR'],axis=1)
    copy_df = pre_df.copy()
    pre_df, df_2 = pre_df.reset_index(), df_2.reset_index()
    pre_df, df_2 = pre_df.drop(['index'],axis=1),df_2.drop(['index'],axis=1)    
        
    # create a boolean mask that identifies NaN values
    nan_mask = df_2.isna() 
    # use np.where to find integer positions of NaN values
    nan_indices = list(np.where(nan_mask))
    scaler = MinMaxScaler(feature_range=(pre_df.index[0], pre_df.index[-1]))
    nan_indices[0] = scaler.fit_transform(nan_indices[0].reshape(-1, 1))
    nan_indices[0] = nan_indices[0].reshape(1,-1)[0].astype(int)
    
    nan_indices = tuple(zip(nan_indices))
    
    pre_df.iloc[nan_indices] = np.nan
    pre_df.index = copy_df.index

#     test_gaps = []
#     for col in range(len(df.columns)):
#         inx_ind = []
#         for index in range(len(df.index)):
#             if index in inx_ind: continue
#             c = 0
#             while np.isnan(df.iloc[index+c,col]) and df.iloc[index+c,col] != df.iloc[-1,col]:
#                 inx_ind += [index+c]
#                 c += 1
#             if not c: continue
#             dt = (df.index[index+c] - df.index[index]).total_seconds()
#             nan_gap = 3600
#             if dt >= nan_gap:
#                 test_gaps += inx_ind
#     if test_gaps:
#         df = df.drop(df.iloc[test_gaps].index, axis=0)
#         copy_df = copy_df.drop(copy_df.iloc[test_gaps].index, axis=0)
    
    pre_df.to_csv('Data_Summary_Files/test_data.csv')
    
    m_values_mask = pre_df.isna()
    
    return pre_df, copy_df, m_values_mask


# Calculating the error between predicted and test values

def error_metric(error_df, copy_df, m_values_mask):
    
    error_dict = {}
    
    for col in error_df.columns:
        test_val = copy_df[col][m_values_mask[col]]
        pred_val = error_df[col][m_values_mask[col]]

        mae = mean_absolute_error(test_val,pred_val)
        mse = mean_squared_error(test_val, pred_val)
        rmse = np.sqrt(mse)
        r2 = r2_score(test_val, pred_val)
        ev = explained_variance_score(test_val, pred_val)
        msle = mean_squared_log_error(test_val, pred_val)
        
        error_dict[col] = [mae,mse,rmse,ev,msle,r2]

        print(f"For {col}")
        print(f"Mean Absolute Error: {mae}")
        print(f"Mean Squared Error: {mse}")
        print(f"Root Mean Squared Error: {rmse}")
        print(f"Explained Variance: {ev}")
        print(f"Mean Squared Log Error: {msle}")
        print(f"R2 Score: {r2} \n")
        
    error_df = pd.DataFrame(error_dict).T
    error_df.columns = ['mae','mse','rmse','ev','msle','r2']
    
    print(error_df)
    
    px.bar(error_df, x = error_df.index, y = error_df['r2']).show()
    
    return error_df


# Main Function, user defines files to be used for performance evaluation refrence documentation for more

def main(year,month,year_2,month_2,file):
    if not re.search(r'\d{4}',year):
        raise Exception(f"Incorret Input: {year}")
    elif not re.search(r'[A-Za-z]{3}',month):
        raise Exception(f"Incorret Input: {month}")
    elif not re.search(r'[A-Za-z]{3}',month_2):
        raise Exception(f"Incorret Input: {month_2}")
    elif not re.search(r'\d{4}',year_2):
        raise Exception(f"Incorret Input: {year_2}")
    elif not [file_i for file_i in ['Irradiance','Deger','Fixed'] if re.search(fr'{file}',file_i)]:
        raise Exception(f"Incorret Input: File")
    else:
        path_list = execute(file)
        if not year and month and year_2 and month_2:
            df_cleaner(path_list,file)
        else:
            path_list = path_func(year,month,year_2,month_2,path_list,file)
            df = df_cleaner([path_list[0]],file)
            df_2 = df_cleaner([path_list[1]],file)
    return df, df_2,file

# Load all the Data
df_load, df2_load, file = main(year = input("Year (format: YYYY): "),month = input("Month (format: jul): "),
     year_2 = input("Second Year (format: YYYY): "),month_2 = input("Second Month (format: jul): "),
     file = input("File (opt: Irradiance/Deger/Fixed): "))

# Clean all the data
df_final, copy_final, m_values_mask = prep(df_load, df2_load)
print(df_final)
if 'DiffuseIR' in df_final.columns:
    px.scatter(df_final,x=df_final.index,y='DirectIR', title = 'Test Dataframe').show()
    px.scatter(copy_final,x=copy_final.index,y='DirectIR', title = 'Observed Dataframe').show()
    
total_nan = df_final.isna().sum().sum()
total_values = df_final.size
mt_count = df_final.isna().all(axis=1).sum()
t_perc = round(total_nan/total_values * 100,3)
mt_perc = round(mt_count*5/total_values * 100,3)

print("Test Dataframe NaN Summary: \n")

print(f"Percentage of NaN values due to System Outage: {mt_perc}% \n")

print(f"Precentage of MAR NaN values: {round(t_perc-mt_perc,3)}% \n")

print(f"Precentage of Total NaN values: {t_perc}% \n")

total_nan = copy_final.isna().sum().sum()
total_values = copy_final.size
mt_count = copy_final.isna().all(axis=1).sum()
t_perc = round(total_nan/total_values * 100,3)
mt_perc = round(mt_count*5/total_values * 100,3)

print("Observed Dataframe NaN Summary: \n")

print(f"Percentage of NaN values due to System Outage: {mt_perc}% \n")

print(f"Precentage of MAR NaN values: {round(t_perc-mt_perc,3)}% \n")

print(f"Precentage of Total NaN values: {t_perc}%")

imputation_dict = {}

# ========================== Imputation Methods, and performance =========================== #

# LOCF

def fill_forward(test_df, copy_final, m_value_mask):
    
    test_df.iloc[[-1,0],:] = 0

    for col in test_df.columns:
        
        if not test_df[col].isna().sum().sum(): continue

        test_df[col] = test_df[col].fillna(method='ffill')
    
    error_df = error_metric(test_df, copy_final, m_values_mask)
    
    if 'DiffuseIR' in test_df.columns:
        px.scatter(test_df,x=test_df.index,y='DirectIR').show()
        
    return error_df
    
imputation_dict['LOCF'] = fill_forward(df_final.copy(), copy_final.copy(), m_values_mask.copy()).to_dict()

# Nearest Value

def nearest(df, copy_final = copy_final.copy(), m_values_mask=m_values_mask.copy()):
    
    df.iloc[[-1,0],:] = 0
    
    for col in df.columns:
        
        if not df[col].isna().sum().sum(): continue

        df[col] = df[col].interpolate(method='nearest')
            
    error_df = error_metric(df, copy_final, m_values_mask)
    
    if 'DiffuseIR' in df.columns:
        px.scatter(df,x=df.index,y='DirectIR').show()
        
    return error_df

imputation_dict['Nearest Neighbor'] = nearest(df_final.copy()).to_dict()

# Linear Interpolation

def interpolate_linear(df, copy_final = copy_final.copy(), m_values_mask=m_values_mask.copy()):
    
    df.iloc[[-1,0],:] = 0
    
    for col in df.columns:
        
        if not df[col].isna().sum().sum(): continue

        df[col] = df[col].interpolate(method='linear')
            
    error_df = error_metric(df, copy_final, m_values_mask)
    
    if 'DiffuseIR' in df.columns:
        px.scatter(df,x=df.index,y='DirectIR').show()
    
    return error_df

imputation_dict['Linear Interpolation'] = interpolate_linear(df_final.copy()).to_dict()

def kalman(df, copy_final = copy_final.copy(), m_values_mask=m_values_mask.copy()):
    
    for col in df.columns:
        
        arr = np.ndarray.tolist(df[col].values)
        arr = robjects.FloatVector(arr)

        df[col] = kalman_StructTs(arr, model = "StructTS")
    
    error_df = error_metric(df, copy_final, m_values_mask)
    
    if 'DiffuseIR' in df.columns:
        px.scatter(df,x=df.index,y='DirectIR').show()
    
    return error_df

imputation_dict['Kalman Smoothing'] = kalman(df_final.copy()).to_dict()

def ARIMA(df, copy_final = copy_final.copy(), m_values_mask=m_values_mask.copy()):
    
    for col in df.columns:
        
        arr = np.ndarray.tolist(df[col].values)
        arr = robjects.FloatVector(arr)

        df[col] = kalman_auto_arima(arr, model = "auto.arima")
    
    error_df = error_metric(df, copy_final, m_values_mask)
    
    if 'DiffuseIR' in df.columns:
        px.scatter(df,x=df.index,y='DirectIR').show()
        
    return error_df

imputation_dict['ARIMA'] = ARIMA(df_final.copy()).to_dict()

def seasonal_decom(df, copy_final = copy_final.copy(), m_values_mask=m_values_mask.copy()):
    
    for col in df.columns:
        
        arr = np.ndarray.tolist(df[col].values)
        arr = robjects.FloatVector(arr)

        df[col] = sea_decom(arr, algorithm = "kalman")
    
    error_df = error_metric(df, copy_final, m_values_mask)
    
    if 'DiffuseIR' in df.columns:
        px.scatter(df,x=df.index,y='DirectIR').show()
        
    return error_df

imputation_dict['Seasonal Decomposition'] = seasonal_decom(df_final.copy()).to_dict()

def knn(df, copy_final = copy_final.copy(), m_values_mask=m_values_mask.copy()):
    
    df['Seconds'] = [(time - time.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() for time in df.index]
    df['Day'] = [d.day for d in df.index]
    
    for col in df.columns:
        if df[col].isna().sum():
            df_KNN = df[[col,'Day', 'Seconds']].copy()
            scaler = MinMaxScaler()
            scaled_df = pd.DataFrame(scaler.fit_transform(df_KNN), columns = df_KNN.columns)
            imputer = KNNImputer(n_neighbors=7,weights='distance')
            knn_solar = pd.DataFrame(imputer.fit_transform(scaled_df),
                                    columns=scaled_df.columns)
            inverse_knn_solar = pd.DataFrame(scaler.inverse_transform(knn_solar),
                                columns=knn_solar.columns, index=df_KNN.index)
            df[col] = inverse_knn_solar[col]
            
    error_df = error_metric(df.drop(['Seconds','Day'], axis = 1), copy_final, m_values_mask)
    
    if 'DiffuseIR' in df.columns:
        px.scatter(df,x=df.index,y='DirectIR').show()
        
    return error_df

imputation_dict['K-Nearest Neighbor'] = knn(df_final.copy()).to_dict()

# ========================= Previous Work ========================== #

def get_int_loc(index,column,df):
    i = df.index.get_loc(index)
    
    if column == 'DiffuseIR':
        c = 0
    elif column == 'DirectIR':
        c = 1
    elif column == 'WindSpeed':
        c = 2
    elif column == 'Temperature':
        c = 3

    return [i,c]    

def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

def get_next_good_index(index,column,df):
    while np.isnan(df.loc[index,column]):
        index.replace(day=(index.day + 1))
        index = nearest(df.index, index)
        if index.day > df.index.day[-1]:
            raise Exception('Out of Bounds Error')
    
    return index

def get_previous_good_index(index,column,df):
    while np.isnan(df.loc[index,column]):
        index.replace(day=(index.day - 1))
        index = nearest(df.index, index)
        if index.day < 1:
            raise Exception('Out of Bounds Error')
        
    return index
    
def small_hole_interpolation(index,column,df):
    [i,c] = get_int_loc(index,column,df)
    lasti = i-1
    while np.isnan(df.iloc[lasti,c]):
        lasti -= 1
    nexti = i+1
    while np.isnan(df.iloc[nexti,c]):
        nexti += 1
    last = df.iloc[lasti,c]
    next = df.iloc[nexti,c]
    dt1 = (df.index[nexti] - df.index[lasti])/np.timedelta64(1,'s')
    m = (next-last)/dt1
    dt2 = (df.index[i] - df.index[lasti])/np.timedelta64(1,'s')
    new_value = last + m*dt2
    return new_value

def surrounding_day_interpolation(index,column,df):
    try:
        last_index = get_previous_good_index(index,column,df)
    except:
        next_index = get_next_good_index(index,column,df)
        return df.loc[next_index,column]
    try:
        next_index = get_next_good_index(index,column,df)
    except:
        last_index = get_previous_good_index(index,column,df)
        return df.loc[last_index,column]
    
    last = df.loc[last_index,column]
    next = df.loc[next_index,column]
    
    dt1 = (next_index - last_index)/np.timedelta64(1,'s')
    m = (next-last)/dt1
    
    dt2 = (index - last_index)/np.timedelta64(1,'s')
    new_value = last + m*dt2
    return new_value

def last_hole(index,column,df):
    [i,c] = get_int_loc(index,column,df)
    
    l = 1
    
    while np.isnan(df.iloc[i+1,c]):
        l = l + 1
        
        if i < len(df):
            i = i + 1
        else:
            break
    
    return [df.index[i],l]

def interpolate(c,df):
    for j in df.index:
        if np.isnan(df.loc[j][c]):
            [i,l] = last_hole(j,c,df)
            dt = (i - j)/np.timedelta64(1,'s')
            if dt <= 10800:
                [start,_] = get_int_loc(j,c,df)
                for x in range(start,start+l):
                    new_value = small_hole_interpolation(df.index[x],c,df)
                    df.at[df.index[x],c] = new_value
            elif dt <= 864000:
                [start,_] = get_int_loc(j,c,df)
                if j.day == 1:
                    for x in range(start,start+l):
                        try:
                            next_index = get_next_good_index(df.index[x],c,df)
                            new_value = df.loc[next_index,c]
                            df.at[df.index[x],c] = new_value
                        except:
                            if c != 'wind_speed':
                                begin = j.round(freq='T')
                                end = i.round(freq='T')
                                bsrn_data = pvlib.iotools.get_bsrn(station='pal',start=begin,end=end,username='bsrnftp',password='bsrn1')
                                [start,_] = get_int_loc(j,c,df)
                                for x in range(start,start+l):
                                        df.at[df.index[x],c] = bsrn_data.loc[nearest(bsrn_data.index,df.index[x]),c]
                                        print('Used BSRN Network')
                                try:
                                    begin = j.round(freq='T')
                                    end = i.round(freq='T')
                                    bsrn_data = pvlib.iotools.get_bsrn(station='pal',start=begin,end=end,username='bsrnftp',password='bsrn1')
                                    [start,_] = get_int_loc(j,c,df)
                                    for x in range(start,start+l):
                                            df.at[df.index[x],c] = bsrn_data.loc[nearest(bsrn_data.index,df.index[x]),c]
                                            print('Used BSRN Network')
                                except:
                                    print('Network blocked connection to BSRN. Large data gap could not be filled.')
                            else:
                                df.at[j:i,c] = 0
                elif i.day == df.index.day[-1]:
                    for x in range(start,start+l):
                        try:
                            last_index = get_previous_good_index(df.index[x],c,df)
                            new_value = df.loc[last_index,c]
                            df.at[df.index[x],c] = new_value
                        except:
                            if c != 'wind_speed':
                                begin = j.round(freq='T')
                                end = i.round(freq='T')
                                bsrn_data = pvlib.iotools.get_bsrn(station='pal',start=begin,end=end,username='bsrnftp',password='bsrn1')
                                [start,_] = get_int_loc(j,c,df)
                                for x in range(start,start+l):
                                        df.at[df.index[x],c] = bsrn_data.loc[nearest(bsrn_data.index,df.index[x]),c]
                                        print('Used BSRN Network')
                                try:
                                    begin = j.round(freq='T')
                                    end = i.round(freq='T')
                                    bsrn_data = pvlib.iotools.get_bsrn(station='pal',start=begin,end=end,username='bsrnftp',password='bsrn1')
                                    [start,_] = get_int_loc(j,c,df)
                                    for x in range(start,start+l):
                                            df.at[df.index[x],c] = bsrn_data.loc[nearest(bsrn_data.index,df.index[x]),c]
                                            print('Used BSRN Network')
                                except:
                                    print('Network blocked connection to BSRN. Large data gap could not be filled.')
                            else:
                                df.at[j:i,c] = 0
                else:
                    for x in range(start,start+l):
                        new_value = surrounding_day_interpolation(df.index[x],c,df)
                        if abs(new_value) >= 0.05:
                            df.at[df.index[x],c] = new_value
                        else:
                            df.at[df.index[x],c] = 0
            else:
                if c != 'wind_speed':
                    try:
                        begin = j.round(freq='T')
                        end = i.round(freq='T')
                        bsrn_data = pvlib.iotools.get_bsrn(station='pal',start=begin,end=end,username='bsrnftp',password='bsrn1')
                        [start,] = get_int_loc(j,c,df)
                        for x in range(start,start+l+1):
                                df.at[df.index[x],c] = bsrn_data.loc[nearest(bsrn_data.index,df.index[x]),c]
                                print('Used BSRN Network')
                    except:
                        print('Network blocked connection to BSRN. Large data gap could not be filled.')
                else:
                    df.at[j:i,c] = 0
    return df[c]

def other_fun(df, copy_final = copy_final.copy(), m_values_mask=m_values_mask.copy()):
    
    for col in df.columns:
        
        if not df[col].isna().sum(): continue
            
        df[col] = interpolate(col,df)
#         if col == 'Temperature':
#             df[col] = df[col].fillna(method='ffill')
#         else:
#             df[col] = interpolate(col,df)
                    
    error_df = error_metric(df, copy_final, m_values_mask)
    
    if 'DiffuseIR' in df.columns:
        px.scatter(df,x=df.index,y='DirectIR').show()
    
    return error_df

imputation_dict['Prior Function'] = other_fun(df_final.copy()).to_dict()

#=========================== Previous Work End ============================ #

def performance(impute_dict):
    imputation_df = pd.DataFrame.from_dict({(outerKey, innerKey): values for outerKey, innerDict in impute_dict.items() for innerKey, values in innerDict.items()}).T
    r2_df = pd.DataFrame(imputation_df.drop(['mae','mse','rmse','ev','msle'], axis=0,level=1).max(axis=0),columns = ['r2'])
    r2_df['Method'] = imputation_df.drop(['mae','mse','rmse','ev','msle'], axis=0,level=1).idxmax(axis=0).values
    mae_df = pd.DataFrame(imputation_df.drop(['r2','mse','rmse','ev','msle'], axis=0,level=1).max(axis=0),columns = ['mae'])
    mae_df['Method'] = imputation_df.drop(['r2','mse','rmse','ev','msle'], axis=0,level=1).idxmax(axis=0).values
    for row in r2_df.index:
        print(f"Optimal Imputation Method for {row}: {r2_df.loc[row]['Method'][0]}, r2 score: {round(r2_df.loc[row]['r2'],5)}")
        print(f"Optimal Imputation Method for {row}: {mae_df.loc[row]['Method'][0]}, mae score: {round(mae_df.loc[row]['mae'],5)}\n")
    return imputation_df
    
performance(imputation_dict.copy())