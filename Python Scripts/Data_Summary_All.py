#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Last updated: on Wednesday Apr 12 16:03 2022

@author: Ethan Masters

Purpose: Data Summary Script For All Years

Python Version: Python 3.9.13 (main, Aug 25 2022, 18:29:29) 
"""

import os
import pandas as pd
import numpy as np
import pvlib
import re

import plotly.express as px

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

# === Df Cleaner Helper Functions === #

def reshape_df(df):
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

    return df

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

# === Df Cleaner Main Function === #

def df_cleaner(df,file):

    # ==== reshaping df for timestap & adjusted headers ==== #
    df = reshape_df(df)

    # === filling gaps in time intervals === #
    df = add_missing_times(df)

    # # ==== Using PvLib to remove nightime values === #
    df = remove_night(df)
    
    if file == 'Irradiance':
        # === Removing Values for Irradiance === #
        df = irr(df)

    else:
        # === Removing Values for Deger & Fixed === #
        df = deg_fix(df)
    
    return df

# === Summary Helper Functions === #

def summarize_NaN(df):
    total_nan = df.isna().sum().sum()
    total_values = df.size
    mt_count = df.isna().all(axis=1).sum()
    # Percentage of total NaN values
    t_perc = round(total_nan/total_values * 100,3)
    # Percentage of NaN values where no timestamp was recorded
    mt_perc = round(mt_count*len(df.columns)/total_values * 100,3)
    
    col_name = []
    col_perc = ()
    
    for col in df.columns:
        n_miss = df[col].isna().sum()
        perc = round(n_miss / total_values * 100,3)
        col_name += [col]
        col_perc += (perc,)

    return t_perc,mt_perc,col_name,col_perc

def NaN_by_month(path_list,file):
    month_data = []
    for path in path_list:
        df = pd.read_csv(path,sep="\t|,",engine='python')
        if df.empty:
            raise Exception(f"The path: {path} loaded an empty dataframe.")
         
        df = df_cleaner(df,file)
        nan_perc,m_perc,col_name,col_perc = summarize_NaN(df)
        df.index = df.index.tz_localize(None)
        month_data += [(df.index[0].to_period('M'),nan_perc,m_perc) + col_perc]
        
    month_data = sorted(month_data, key = lambda x : x[1])
    return pd.DataFrame(month_data, columns = ['Month','Total NaN (%)','System Outage NaN (%)'] + col_name).set_index('Month')

# === Update & Main Script Function === #

def update(file):
    # === path collection start === #
    path_list = execute(file)

    # === finding month with least NaN values === ###
    update_df = NaN_by_month(path_list,file)
    path = re.sub(r'Notebooks|Python Scripts','Support Files/',os.getcwd())
    update_df.to_csv(path + f'{file}_NaN_All.csv')
    main(file)

def main(file):
    path = re.sub(r'Notebooks|Python Scripts','Support Files/',os.getcwd())
    file_data = pd.read_csv(path + f'{file}_NaN_All.csv', index_col= 'Month').sort_index()
    col = [col for col in file_data.columns if col != 'Month']
    fig = px.line(file_data, x=file_data.index, y=col, hover_name = 'Month', title=f"{file}: Percentage of NaN by Month")
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")])))
    fig.show()
    
file = input("File (opt: Irradiance/Deger/Fixed): ")
response = input("Last update: April 1st 2023 \n To continue press: 'Enter' \n Else type: 'update()' \n\t")

if response == 'update()':
    update(file)
else:
    main(file)