#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Last updated: on Wednesday Apr 12 16:03 2022

@author: Ethan Masters

Purpose: Compile All Data Type into one csv file

Python Version: Python 3.9.13 (main, Aug 25 2022, 18:29:29) 
"""

import os
import pandas as pd
import numpy as np
import pvlib
import re

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

    if df.index[0].year < 2021:
        seconds_gap = 240
        frequency = '120s'
    elif df.index[0].year >= 2021:
        seconds_gap = 22
        frequency = '11s'
    
    # creating of list of times to find interval gaps
    time_list = list(df.index)
    
    # calculating interval gaps if > 21s and storing [interval length (s), start_time, end_time]
    missing_intervals = [[(time_list[time+1] - time_list[time]).total_seconds(),time_list[time],time_list[time+1]]
                 for time in range(len(time_list)-1) if (time_list[time+1] - time_list[time]).total_seconds() > seconds_gap]
    # generating time stamps to fill interval gaps 
    interval_list = [element for sublist in [pd.date_range(start=interval[1],
                             end=interval[2]-pd.Timedelta(1,'s'),
                             freq=frequency) for interval in missing_intervals] for element in sublist]
    
    # checking for missing values at the beginning of the month
    if time_list[0] > time_list[0].replace(day=1,hour=1):
#         print("Found a month that has missing values in the beginning of the month.")
#         print('Time:',time_list[0])
        interval_list += [time for time in pd.date_range(start=time_list[0].replace(day=1,hour=0,minute=0,second=0),
                             end=time_list[0]-pd.Timedelta(1,'s'),
                             freq=frequency)]
        missing_intervals += [[(time_list[0] - time_list[0].replace(day=1,hour=0,minute=0,second=0)).total_seconds(),
                             time_list[0].replace(day=1,hour=0,minute=0,second=0),time_list[0]]]
        
    # checking for missing values at the end of the month    
    next_month = time_list[0].replace(day=28,hour=0,minute=0,second=0) + pd.Timedelta(4,'d')
    last_day = next_month - pd.Timedelta(next_month.day,'d')
    if time_list[-1] < last_day.replace(hour = 23,minute=0):
#         print("Found a month that has missing values in the end of the month.")
#         print('Time:',time_list[-1])
        interval_list += [time for time in pd.date_range(start=time_list[-1],
                     end=last_day.replace(hour=23,minute=59,second=59),
                     freq=frequency)]
        missing_intervals += [[(last_day.replace(hour=23,minute=59,second=59) - time_list[-1]).total_seconds(),
                             time_list[-1],last_day.replace(hour=23,minute=59,second=59)]]
        
    interval_list = list(set(interval_list))
    mt_df = pd.DataFrame(index=interval_list,columns=df.columns)
    mt_df.loc[interval_list] = np.nan
    df = pd.concat([df,mt_df], axis = 0).sort_index()

    return df

def time_features(df):
    df['day'] = [d.day for d in df.index]
    df['month'] = [d.month for d in df.index]
    df['year'] = [d.year for d in df.index]
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


def df_cleaning(path_list,file):
    
    df = pd.DataFrame()

    for path in path_list:

        df_load = pd.read_csv(path,sep="\t|,",engine='python')
        
        if df_load.empty:
            raise Exception(f"The path: {path} loaded an empty dataframe.")
        
        # ==== reshaping df for timestap & adjusted headers ==== #
        df_load = reshape_df(df_load,file)
        
        # === filling gaps in time intervals === #
        df_load= add_missing_times(df_load)

        # # ==== Using PvLib to remove nightime values === #
        df_load = remove_night(df_load)
        
        if file == 'Irradiance':
            df_load = irr(df_load)
        else:
            df_load = deg_fix(df_load)
        
        df = pd.concat([df,df_load],axis=0,ignore_index=False).sort_index()
    
    return df


def main(file):
    if not [file_i for file_i in ['Irradiance','Deger','Fixed'] if re.search(fr'{file}',file_i)]:
            raise Exception(f"Incorret Input: File")
    path_list = execute(file)
    df = df_cleaning(path_list,file)
    datapath = re.sub(r'Notebooks|Python Scripts','Support Files/',os.getcwd())
    df.to_csv(datapath + 'Irradiance.csv')

main(file = input("File (opt: Irradiance/Deger/Fixed): "))

