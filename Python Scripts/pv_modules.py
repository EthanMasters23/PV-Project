#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Last updated: on Wednesday Apr 12 16:03 2022

@author: Ethan Masters

Purpose: PV Modules

Python Version: Python 3.9.13 (main, Aug 25 2022, 18:29:29) 
"""

import os
import pandas as pd
import numpy as np
import pvlib
import re

def get_file_paths(file):

    """
    Retrieves the file paths for a given file name from the data directory.

    Args:
        file (str): Name of the file to search for.

    Returns:
        list: List of file paths matching the given file name.

    """
    
    path_list = []
    datapath = re.sub(r'Notebooks|Python Scripts','Data/',os.getcwd())
    for dir in os.scandir(datapath):
        if re.search(r'\.',dir.name): continue
        year_path = datapath + f"{dir.name}"
        for dir in os.scandir(year_path):
            if dir.name == file:
                month_path = year_path + f"/{dir.name}/"
                for dir in os.scandir(month_path):
                    if not re.search(r'\.csv|\.xlsx',dir.name) or re.search(r'clean_|test',dir.name): continue
                    path_list += [month_path + f"{dir.name}"]
    return path_list

def path_function(year,month,path_list,file):

    """
    Description: Searches for a specific path within the provided path_list based on the given year and month parameters.
    Arguments:
        - year (str): The year to search for.
        - month (str): The month to search for.
        - path_list (list): A list of paths to search within.
        - file (str): The file for which the search is being performed.
    Returns:
        - path_found (list): A list containing the path(s) that match the specified year and month.
    """

    path_found = []
    for path in path_list:
        y,m = re.search(r"/(\d{4})/[a-z]*/([a-z]*)\.",path.lower()).group(1,2)
        if re.search(fr"{year}",y) and re.search(fr"{month}",m):
            path_found += [path]
            break

    print(f"\nSummary of data for {file} in {m}, {y} \n")
    return path_found

def path_function_extended(year,month,year_2,month_2,path_list):
    
    """
    Description: Searches for specific paths within the provided path_list based on the given year and month parameters.
    If year and month are provided, it searches for a single path that matches the year and month.
    If year_2 and month_2 are provided, it searches for paths that match either year and month or year_2 and month_2.
    Arguments:
        - year (str): The year of the first file to search for.
        - month (str): The month of the first file to search for.
        - year_2 (str): The year of the second file to search for.
        - month_2 (str): The month of the second file to search for.
        - path_list (list): A list of paths to search within.
        - file (str): The file for which the search is being performed.
    Returns:
        - path_found (list): A list containing the path(s) that match the specified year and month parameters.
    """

    path_found = []
    for path in path_list:
        path_copy = path.lower()
        data = re.search(r"/(\d{4})/[a-z]*/([a-z]*)\.",path_copy).group(1,2)
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

    return path_found


def reshape_df(df, file):
    
    """
    Reshapes the input DataFrame by manipulating the 'DayID' and 'TimeID' columns.

    Args:
        df (pandas.DataFrame): Input DataFrame containing 'DayID' and 'TimeID' columns.

    Returns:
        pandas.DataFrame: Reshaped DataFrame with the 'date' column as the index, localized to 'Etc/UTC' timezone.

    """
    
    # Convert 'DayID' and 'TimeID' columns to strings
    df['DayID'] = df['DayID'].astype(str)
    df['TimeID'] = df['TimeID'].astype(str)

    # Create 'date' column by combining 'DayID' and 'TimeID'
    df['index'] = df['DayID'] + 'T' + df['TimeID']

    # Drop unnecessary columns
    df = df.drop(columns=['DayID', 'TimeID'])

    # Convert 'date' column to datetime
    df['index'] = pd.to_datetime(df['index'])

    # Set 'date' column as the index
    df = df.set_index('index')

    # Localize index timezone to 'Etc/UTC'
    df.index = df.index.tz_localize(tz='Etc/UTC')

    # Sort the DataFrame by index
    df = df.sort_index()

    if file == 'Irradiance':
        df.columns = ['GlobalIR','DirectIR','DiffuseIR','WindSpeed','Temperature']
    else:
        df.columns = ['MonoSi_Vin','MonoSi_Iin','MonoSi_Vout','MonoSi_Iout','PolySi_Vin','PolySi_Iin','PolySi_Vout','PolySi_Iout','TFSi_a_Vin','TFSi_a_Iin','TFSi_a_Vout','TFSi_a_Iout','TFcigs_Vin','TFcigs_Iin','TFcigs_Vout','TFcigs_Iout','TempF_Mono','TempF_Poly','TempF_Amor','TempF_Cigs']

    return df

def add_missing_times(df):
        
    """
    Adds missing times to a DataFrame by identifying interval gaps and filling them with timestamp values.

    Args:
        df (pandas.DataFrame): Input DataFrame containing a datetime index.

    Returns:
        pandas.DataFrame: DataFrame with missing times added and sorted index.

    """

    if df.index[0].year < 2021:
        seconds_gap = 240
        frequency = '120s'
    elif df.index[0].year >= 2021:
        seconds_gap = 30
        frequency = '15s'
    
    # creating of list of times to find interval gaps
    time_list = list(df.index)
    
    # calculating interval gaps if > 21s and storing [interval length (s), start_time, end_time]
    missing_intervals = [[(time_list[time+1] - time_list[time]).total_seconds(),time_list[time],time_list[time+1]]
                 for time in range(len(time_list)-1) if (time_list[time+1] - time_list[time]).total_seconds() > seconds_gap]
    # generating time stamps to fill interval gaps 
    interval_list = [element for sublist in [pd.date_range(start=interval[1],
                             end=interval[2]-pd.Timedelta(1,'s'),
                             freq=frequency) for interval in missing_intervals] for element in sublist]
    
    # = consider for bigO notation = #
    # import itertools
    # interval_list = list(itertools.chain.from_iterable(
    #     pd.date_range(start=interval[1], end=interval[2]-pd.Timedelta(1, 's'), freq=frequency)
    #     for interval in missing_intervals
    # ))
    # = #

    # checking for missing values at the beginning of the month
    if time_list[0] > time_list[0].replace(day=1,hour=4):
        # print("Found a month that has missing values in the beginning of the month.")
        # print('Time:',time_list[0])
        interval_list += [time for time in pd.date_range(start=time_list[0].replace(day=1,hour=0,minute=0,second=0),
                             end=time_list[0]-pd.Timedelta(1,'s'),
                             freq='11s')]
        missing_intervals += [[(time_list[0] - time_list[0].replace(day=1,hour=0,minute=0,second=0)).total_seconds(),
                             time_list[0].replace(day=1,hour=0,minute=0,second=0),time_list[0]]]
        
    # checking for missing values at the end of the month    
    next_month = time_list[0].replace(day=28,hour=0,minute=0,second=0) + pd.Timedelta(4,'d')
    last_day = next_month - pd.Timedelta(next_month.day,'d')
    if time_list[-1] < last_day.replace(hour = 23,minute=0):
        # print("Found a month that has missing values in the end of the month.")
        # print('Time:',time_list[-1])
        interval_list += [time for time in pd.date_range(start=time_list[-1],
                     end=last_day.replace(hour=23,minute=59,second=59),
                     freq='11s')]
        missing_intervals += [[(last_day.replace(hour=23,minute=59,second=59) - time_list[-1]).total_seconds(),
                             time_list[-1],last_day.replace(hour=23,minute=59,second=59)]]
        
    interval_list = list(set(interval_list))
    mt_df = pd.DataFrame(index=interval_list,columns=df.columns)
    mt_df.loc[interval_list] = np.nan
    df = pd.concat([df,mt_df], axis = 0).sort_index()

    return df, missing_intervals

def remove_night(df):
    """
    Removes nighttime data from a DataFrame based on solar position information.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with nighttime data removed.

    """
    lat = 49.102
    lon = 6.215
    alt = 220
    solpos = pvlib.solarposition.get_solarposition(
        time=df.index,latitude=lat,longitude=lon,altitude=alt,method='pyephem')
    df = df[solpos['zenith'] <=90]
    return df

def clean_irradiance_values(df):
    
    """
    Cleans irradiance values in a DataFrame by removing outliers and negative values.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with cleaned irradiance values.

    """
    
    # Removing Temperature Values #
    df[(df['Temperature'] > 60) | (df['Temperature'] < 0.1)] = np.nan

    # Removing Wind Speed Values #
    df[df['WindSpeed'] > 100] = np.nan

    # Removing DirectIR Values #
    df[df['DirectIR'] > 2000] = np.nan

    # Removing DiffuseIR Values #
    df[df['DiffuseIR'] > 2000] = np.nan

    # Removing Negative Values #
    df[df < 0] = np.nan

    return df

def clean_deger_fixed_values(df):
    
    """
    Cleans deger & fixed values in a DataFrame by removing negative values.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with cleaned irradiance values.
        
    """
    
    # Removing Negative Values #
    df[df < 0] = np.nan

    return df

def time_features(df):

    """
    Extracts day, month, and year features from the index of a DataFrame.

    Args:
        df (DataFrame): Input DataFrame containing a datetime index.

    Returns:
        DataFrame: Modified DataFrame with 'day', 'month', and 'year' columns added.
    """

    df['day'] = [d.day for d in df.index]
    df['month'] = [d.month for d in df.index]
    df['year'] = [d.year for d in df.index]

    return df