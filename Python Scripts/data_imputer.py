#!/usr/bin/env python
# coding: utf-8

# $$\large \text{Packages & Specs} $$

# In[ ]:


import os
import pandas as pd
import numpy as np
import re
import threading
import queue

# Kalman Smoothing using R objects
import rpy2.robjects as robjects
# import R packages
from rpy2.robjects.packages import importr

# Impute TS
imputeTS = importr('imputeTS')
kalman_StructTs = robjects.r['na_kalman']

import sys

module_path = re.sub(r'Notebooks','Python Scripts',os.getcwd())
sys.path.append(module_path)
from pv_modules import *


# In[ ]:


def interpolation_method(df, nan_gaps):
    
    """
    Performs interpolation on a DataFrame to fill missing values using the 'time' method.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        nan_gaps (dict): Dictionary containing column names as keys and lists of NaN gap indices as values.

    Returns:
        pandas.DataFrame: DataFrame with filled values using time based linear interpolation.
        
    """
    
    output_df = df.copy()
        
    for col in df.columns:
        
        if not df[col].isna().sum().sum(): continue

        df[col] = df[col].interpolate(method='time', limit_direction='both')
            
    for col in nan_gaps.keys():
        output_df[col].iloc[nan_gaps[col]] = df[col].iloc[nan_gaps[col]]
            
    
    return output_df


# In[ ]:


def ARIMA(df, nan_gaps):
    
    """
    Applies Kalman filtering to a DataFrame to fill missing values.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        nan_gaps (dict): Dictionary containing column names as keys and lists of NaN gap indices as values.

    Returns:
        pandas.DataFrame: DataFrame with filled values using Kalman filtering.

    """
    
    output_df = df.copy()
    
    for col in df.columns:
        
        arr = np.ndarray.tolist(df[col].values)
        arr = robjects.FloatVector(arr)

        df[col] = kalman_StructTs(arr, model = "auto.arima")
        
    for col in nan_gaps.keys():
        output_df[col].iloc[nan_gaps[col]] = df[col].iloc[nan_gaps[col]]
        
    return output_df


# In[ ]:


def df_imputer(df):
    
    """
    Finds the index positions of gaps in a DataFrame based on their size and applies appropriate imputation method.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with filled missing values using interpolation and Kalman filtering.

    """
    
    interpolation = {}
    arima = {}
    
    for col in range(len(df.columns)):
        index_list = []
        interpolation[df.columns[col]] = []
        arima[df.columns[col]] = []
        for index in range(len(df.index)):
            if index in index_list: continue
            day = df.index[index].day
            c = 0
            while np.isnan(df.iloc[index+c,col]) and df.index[index+c].day == day:
                if df.index[index+c] == df.index[-1]: break
                index_list += [index+c]           
                c += 1     
            if not c and not np.isnan(df.iloc[index+c,col]): continue
            dt = (df.index[index+c] - df.index[index]).total_seconds()
            if dt <= 200:
                interpolation[df.columns[col]] += list(range(index,index+c+1))
            else:
                arima[df.columns[col]] += list(range(index,index+c+1))

        if not interpolation[df.columns[col]]:
            del interpolation[df.columns[col]]
        if not arima[df.columns[col]]:
            del arima[df.columns[col]]
            
    if interpolation:
        df = interpolation_method(df,interpolation)
    if arima:
        df = ARIMA(df,arima)
        
    return df


# $$\large \text{Imputer; instance of df for cleaning and preprocessing} $$

# In[ ]:


class Imputer():
    
    """
    Class for data imputation and cleaning.
    
    Attributes:
        df (pandas.DataFrame): Input DataFrame.
        month (str): Month.
        year (str): Year.
        file (str): File type.

    Methods:
        run(): Runs the data imputation and cleaning process.
    """
    
    def __init__(self,df,month,year,file):
        self.df = df
        self.month = month
        self.year = year
        self.file = file
        super().__init__()
    def run(self):
                
        # === reshaping df for timestap & adjusted headers === #
        self.df = reshape_df(self.df,self.file)
        
        # === filling gaps in time intervals === #
        self.df,_ = add_missing_times(self.df)
        
        # === Using PvLib to remove nightime values === #
        self.df = remove_night(self.df)
        
        if self.file == 'Irradiance':
            
            # === Set Column Names === #
            self.df.columns = ['GlobalIR','DirectIR','DiffuseIR','WindSpeed','Temperature']
            
            # === Removing Misread Vemps === #
            self.df = clean_irradiance_values(self.df)
            
        else:
            
            # === Set Column Names === #
            self.df.columns = ['MonoSi_Vin','MonoSi_Iin','MonoSi_Vout','MonoSi_Iout','PolySi_Vin','PolySi_Iin','PolySi_Vout','PolySi_Iout','TFSi_a_Vin','TFSi_a_Iin','TFSi_a_Vout','TFSi_a_Iout','TFcigs_Vin','TFcigs_Iin','TFcigs_Vout','TFcigs_Iout','TempF_Mono','TempF_Poly','TempF_Amor','TempF_Cigs']
        
            # === Removing Misread Values === #
            self.df = clean_deger_fixed_values(self.df)
            
        print(f"Imputing {round(self.df.isna().sum().sum()/self.df.size*100,3)}% of the data for {self.month}, {self.year}.")

        self.df = df_imputer(self.df)
        
        if self.df.isna().any().any():   
            raise Exception(f"The File {self.file}, {self.month} {self.year} still has NaN values")
            
        cwd = re.sub("Notebooks|Python Scripts","Data/",os.getcwd())
        datapath = cwd + self.year + '/' + self.file + '/'
        file = self.month.lower() + '.csv'
        self.df.to_csv(datapath + "/clean_" + file)


# In[ ]:


class Worker(threading.Thread):
    
    """
    Thread worker class for parallel processing.
    
    Attributes:
        queue (Queue): Queue containing file paths.
        file (str): File type.
        lock (threading.Lock): Lock for thread synchronization.
    """
    
    def __init__(self, queue, file, lock):
        threading.Thread.__init__(self)
        self.queue = queue
        self.file = file
        self.lock = lock

    def run(self):
        while True:
            try:
                file_path = self.queue.get(timeout=3) # retrieve file path from the queue
            except queue.Empty:
                return # If the queue is empty, exit the thread
            
            data = re.search(r"/(\d{4})/[a-zA-Z]*/([a-zA-Z]*)\.csv",file_path).group(1,2)
            df = pd.read_csv(file_path, sep="\t|,", engine='python')
            self.lock.acquire()
            print('Starting',data[1], data[0])
            self.lock.release()
            Imputer(df, data[1], data[0], self.file).run()
            self.lock.acquire()
            print('Completed',data[1], data[0])
            self.lock.release()
            self.queue.task_done() # Notify the queue that the task is done


# In[ ]:


starttime = pd.Timestamp.now()
        
q = queue.Queue()

file = input("File (opt: Irradiance/Deger/Fixed): ")
             
file_paths = get_file_paths(file) 

for file_path in file_paths:
    q.put_nowait(file_path)
    # = run for specific month and year, or for test file = #
#     if re.search(r'jul',file_path.lower()) and re.search(r'2022',file_path): 
#         q.put_nowait(file_path)
    
lock = threading.Lock()

for _ in range(6): 
    t = Worker(q, file, lock)
    t.daemon = True
    t.start()

q.join() 

endtime = pd.Timestamp.now()

runtime = endtime - starttime

print("Start:",starttime,"\nEnd:",endtime,"\nRun Time:",runtime.total_seconds())

