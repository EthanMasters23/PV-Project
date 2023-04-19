#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Last updated: on Wednesday Apr 18 17:03 2022

@author: Ethan Masters

Purpose: Data Summary Script

Python Version: Python 3.9.13 (main, Aug 25 2022, 18:29:29) 
"""

import os
import pandas as pd
import numpy as np
import pvlib
import re
import json
import plotly.express as px
import scipy

# === Path Collection Functions === #

def execute(file):
    path_list = []
    datapath = re.sub(r'Notebooks|Python Scripts','Data/',os.getcwd())
    for dir in os.scandir(datapath):
        if re.search(r'\.',dir.name): continue
        year_path = datapath + f"{dir.name}"
        for dir in os.scandir(year_path):
            if dir.name == file:
                month_path = year_path + f"/{dir.name}/"
                for dir in os.scandir(month_path):
                    if not re.search(r'\.csv|\.xlsx',dir.name): continue
                    path_list += [month_path + f"{dir.name}"]
    return path_list

def path_func(year,month,path_list,file):
    path_found = []
    for path in path_list:
        y,m = re.search(r"/(\d{4})/[a-z]*/([a-z]*)\.",path.lower()).group(1,2)
        if re.search(fr"{year}",y) and re.search(fr"{month}",m):
            path_found += [path]
            break

    print(f"Summary of data for {file} in {m}, {y} \n")
    return path_found

# === Df Cleaner Helper Functions === #

def reshape_df(df,file):
    df['DayID'] = df['DayID'].astype(str)
    df['TimeID'] = df['TimeID'].astype(str)
    df['date'] = df['DayID'] + 'T' +  df['TimeID']
    df = df.drop(columns = ['DayID','TimeID'])
    df.date = pd.to_datetime(df.date)
    df = df.set_index('date')
    df.index = df.index.tz_localize(tz = 'Etc/UTC')
    df = df.sort_index()
    if file == 'Irradiance':
        df.columns = ['GlobalIR','DirectIR','DiffuseIR','WindSpeed','Temperature']
    else:
        df.columns = ['MonoSi_Vin','MonoSi_Iin','MonoSi_Vout','MonoSi_Iout','PolySi_Vin','PolySi_Iin','PolySi_Vout','PolySi_Iout','TFSi_a_Vin','TFSi_a_Iin','TFSi_a_Vout','TFSi_a_Iout','TFcigs_Vin','TFcigs_Iin','TFcigs_Vout','TFcigs_Iout','TempF_Mono','TempF_Poly','TempF_Amor','TempF_Cigs']
    return df

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
    
    # checking for missing values at the beginning of the month
    if time_list[0] > time_list[0].replace(day=1,hour=1):
        print("Found a month that has missing values in the beginning of the month.")
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
        print("Found a month that has missing values in the end of the month.")
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

    return df,missing_intervals

def remove_night(df):
    lat = 49.102
    lon = 6.215
    alt = 220
    solpos = pvlib.solarposition.get_solarposition(
        time=df.index,latitude=lat,longitude=lon,altitude=alt,method='pyephem')
    df = df[solpos['zenith'] <= 90]
    return df

def irr(df):
    # Removing Temperature Values #
    df[df['Temperature'] > 60] = np.nan

    # Removing Wind Speed Values #
    df[df['WindSpeed'] > 100] = np.nan

    # Removing DirectIR Values #
    df[df['DirectIR'] > 2000] = np.nan

    # Removing DiffuseIR Values #
    df[df['DiffuseIR'] > 2000] = np.nan

    # Removing Negative Values #
    df[df < 0] = np.nan

    return df

def deg_fix(df):
    # Removing Negative Values #
    df[df < 0] = np.nan

    return df

def time_features(df):
    df['day'] = [d.day for d in df.index]
    df['month'] = [d.month for d in df.index]
    df['year'] = [d.year for d in df.index]
    return df

# === Df Cleaner Main Function === #

def df_cleaning(path_list,file):
    
    df = pd.DataFrame()
    outlier_output = pd.DataFrame()

    missing_intervals = []

    for path in path_list:

        df_load = pd.read_csv(path,sep="\t|,",engine='python')
        
        if df_load.empty:
            raise Exception(f"The path: {path} loaded an empty dataframe.")
        
        # ==== reshaping df for timestap & adjusted headers ==== #
        df_load = reshape_df(df_load,file)
        
        # === copy df for outlier pre-processed === #
        outlier_output = pd.concat([outlier_output,df.copy()],axis=0,ignore_index=False)

        # === filling gaps in time intervals === #
        df_load,m_intervals = add_missing_times(df_load)

        # === Time Features === #
        df_load = time_features(df_load)

        # # ==== Using PvLib to remove nightime values === #
        df_load = remove_night(df_load)
        
        if file == 'Irradiance':
            df_load = irr(df_load)
        else:
            df_load = deg_fix(df_load)
        
        df = pd.concat([df,df_load],axis=0,ignore_index=False).sort_index()
        missing_intervals += m_intervals
    
    return df, missing_intervals, outlier_output

# === Summary Helper Functions === #

def summarize_nan(df):
    total_nan = df.drop(['day','month','year'],axis=1).isna().sum().sum()
    total_values = df.drop(['day','month','year'],axis=1).size
    mt_count = df.drop(['day','month','year'],axis=1).isna().all(axis=1).sum()
    t_perc = round(total_nan/total_values * 100,3)
    mt_perc = round(mt_count*len(df.columns)/total_values * 100,3)

    print(f"Percentage of NaN values due to System Outage: {mt_perc}% \n")
    
    print(f"Precentage of MAR NaN values: {round(t_perc-mt_perc,3)}% \n")

    print(f"Precentage of Total NaN values: {t_perc}%")

    print("\n Missing values by column")

    for col in df.columns:
        if not col in ['day','month','year']:
            n_miss = df[col].isna().sum()
            perc = round(n_miss / df.shape[0] * 100,3)
            print(f"{col}, Missing: {n_miss} ({perc}%)")

    print("\n Missing values by day")

    for row in df['day'].unique():
        n_miss = df[df['day']==row].drop(['day','month','year'],axis=1).isna().sum().sum()
        perc = round(n_miss / df[df['day']==row].drop(['day','month','year'],axis=1).size * 100,3)
        print(f"{row}, Missing: {n_miss} ({perc}%)")

    print("\n Missing values by month")    


    for row in sorted(df['month'].unique()):
        if len(df['month'].unique()) == 1: break
        n_miss = df[df['month']==row].drop(['day','month','year'],axis=1).isna().sum().sum()
        perc = round(n_miss / df[df['month']==row].drop(['day','month','year'],axis=1).size * 100,3)
        print(f"{row}, Missing: {n_miss} ({perc}%)")

    print("\n Missing values by year")    


    for row in df['year'].unique():
        if len(df['year'].unique()) == 1: break
        n_miss = df[df['year']==row].drop(['day','month','year'],axis=1).isna().sum().sum()
        perc = round(n_miss / df[df['year']==row].drop(['day','month','year'],axis=1).size * 100,3)
        print(f"{row}, Missing: {n_miss} ({perc}%) \n")

def mt_fig(missing_intervals):
    interval_df = pd.DataFrame(missing_intervals, columns=['seconds','start_time','end_time'])
    px.scatter(interval_df, x='start_time', y='seconds', hover_data=['start_time','end_time'],title='Intervals in Time of Missing Observations').show()

def col_fig(df):
    for col in df.drop(['day','month','year'],axis=1).columns:
        px.scatter(df, x=df.index, y=f'{col}',title=f'{col}').show()

def timestamps_fig(df):
    df['Seconds'] = [(time - time.replace(hour=0, minute=0,
                                          second=0, microsecond=0)).total_seconds()for time in df.index]
    df = df.dropna(axis=0)   
    px.scatter(df, y='Seconds').show()

def corr_matrix(df):
    df = df.drop(['day','month','year'],axis=1)
    df = df.dropna()
    
    corrs = []
    p_values = []
    
    for feat1 in df.columns:
        corr_list = []
        p_list = []
        for feat2 in df.columns:
            corr, p_value = scipy.stats.spearmanr(df[feat1], df[feat2])
            corr_list += [corr]
            p_list += [p_value]
        corrs += [corr_list]
        p_values += [p_list]
        
    corr_matrix = pd.DataFrame(corrs, index = df.columns, columns = df.columns)
    px.imshow(corr_matrix,text_auto=True,title="Correlation Matrix").show()
    
    p_matrix = pd.DataFrame(p_values, index = df.columns, columns = df.columns)
    px.imshow(p_matrix,text_auto=True,title="P Value Matrix").show()

def outliers(df,pre):
    indicator = True
    if pre: df = df.drop(['day','month','year'],axis=1)
    for col in df.columns:
        arr = df[col]
        z_scores = np.abs((arr - arr.mean()) / arr.std())
        threshold = 3
        outliers = arr[z_scores > threshold]
        if len(outliers):
            indicator = False
            print(f"{col} number of outliers {len(outliers)}, min: {min(outliers)}, max: {max(outliers)} \n")
    if indicator: print("There were no outliers found pre-processing. \n")

def nan_gaps(df):
    nan_gaps = []
    for col in range(len(df.columns)):
        inx_ind = []
        for index in range(len(df.index)):
            if index in inx_ind: continue
            c = 0
            while np.isnan(df.iloc[index+c,col]) and df.iloc[index+c].name != df.iloc[-1].name:
                inx_ind += [index+c]
                c += 1
            if not c: continue
            dt = (df.index[index+c] - df.index[index]).total_seconds()
            nan_gaps += [[dt, df.index[index], df.index[index+c], col]]
    nan_df = pd.DataFrame(nan_gaps, columns = ['Seconds', 'Start Time', 'End Time', 'Column'])
    px.scatter(nan_df, x = nan_df.index, y = 'Seconds', hover_data = ['Start Time','End Time', 'Column']
                     ,title = 'Scatter plot of the NaN gaps (in seconds) over time:').show()
    
# === Summary Main Function === #

def summary(path_list,file,update=False):
    
    # removing and preprocessing df
    if update:
        datapath = re.sub(r'Notebooks|Python Scripts','Support Files/',os.getcwd())
        df = pd.read_csv(datapath + f'{file}_Dataframe.csv', index_col='Unnamed: 0')
        df.index = pd.to_datetime(df.index)
        outlier_output = pd.read_csv(datapath + f'{file}_Outlier_Dataframe.csv', index_col='Unnamed: 0')
        with open(datapath + f"mi_{file}.json", "r") as read_file:
            missing_intervals = json.load(read_file)
        missing_intervals = [[interval[0],pd.to_datetime(interval[1]),
                              pd.to_datetime(interval[2])] for interval in missing_intervals]
    else:
        df, missing_intervals, outlier_output = df_cleaning(path_list,file)
    
    # printing Summary of NaN Values
    print("\nSummary of NaN Values")
    summarize_nan(df)

    # printing outliers pre-processing
    print("Summary of outliers (if any) for pre-processed data:")
    outliers(outlier_output,pre = False)

    # printing outliers post-processing
    print("Summary of outliers (if any) for post-processed data:")
    outliers(df,pre = True)
    
    # Figure of nan gaps
    # too computational heavy
    # nan_gaps(df)

    # figure of missing intverals 
    mt_fig(missing_intervals)

    # figure's of all variables plotted over time
    col_fig(df)
    
    # figure of correlation matrix
    corr_matrix(df)

# === Update & Main Script Function === #

def update(path_list, file):
    datapath = re.sub(r'Notebooks|Python Scripts','Support Files/',os.getcwd())
    df, missing_intervals, outlier_output = df_cleaning(path_list,file)
    outlier_output.to_csv(datapath + fr'{file}_Outlier_Dataframe.csv')
    df.to_csv(datapath + fr'{file}_Dataframe.csv')
    missing_intervals = [[interval[0],str(interval[1]),str(interval[2])] for interval in missing_intervals]
    with open(datapath + fr"mi_{file}.json", "w") as write_file:
        json.dump(missing_intervals, write_file)

def main(year,month,file):
    if not [file_i for file_i in ['Irradiance','Deger','Fixed'] if re.search(fr'{file}',file_i)]:
            raise Exception(f"Incorret Input: File")
    elif not year and not month:
        path_list = execute(file)
        response = input("Last update: April 1st 2023 \n To continue press: 'Enter' \n Else type: 'update()' \n\t")
        if not response:
            summary(path_list,file,update=False)
        elif response == "update()":
            update(path_list, file)
            summary(path_list,file,update=False)
    elif not re.search(r'\d{4}',year):
        raise Exception(f"Incorret Input: Year")
    elif not re.search(r'[A-Za-z]{3}',month):
        raise Exception(f"Incorret Input: Month")
    else:
        path_list = execute(file)
        path_list = path_func(year,month,path_list,file)
        summary(path_list,file)

main(year = input("Year (format: YYYY): "), month = input("Month (format: jul): "),
     file = input("File (opt: Irradiance/Deger/Fixed): "))