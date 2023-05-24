#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Last updated: on Wednesday Apr 12 16:03 2022

@author: Ethan Masters

Purpose: Preprocesses Neural Network Training Data

Python Version: Python 3.9.13 (main, Aug 25 2022, 18:29:29) 
"""

import os
import pandas as pd
import numpy as np
import re

def reshape(df,ml_df):
    ml_df.columns = ml_df.loc['variable',:] + " (" + ml_df.loc['unit',:] + ") (" + ml_df.loc['level',:] + ")"
    ml_df = ml_df.drop(['lat','lon','asl','variable','unit','level','resolution','aggregation','timestamp'],axis=0)
    ml_df.index.name = 'Timestamp'
    ml_df.index = pd.to_datetime(ml_df.index)
    ml_df = ml_df.loc[df.index[0].tz_convert(None).to_period('H').to_timestamp():df.index[-1].tz_convert(None).to_period('H').to_timestamp(),:]
    return ml_df

def resample(df,ml_df): 
    output = pd.DataFrame(index=df.index,columns = ml_df.columns)
    for index in df.index:
        output.loc[index] = ml_df.loc[index.tz_convert(None).to_period('H').to_timestamp()].values
    return output

def execute():

    datapath = re.sub(r'Notebooks|Python Scripts','Support Files/',os.getcwd())

    # == Load Irradiance Data == #
    df = pd.read_csv(datapath + 'Irradiance.csv',index_col=0)
    df.index = pd.to_datetime(df.index)

    # == Load Training Data == #
    ml_df = pd.read_csv(datapath + 'ml_features.csv',index_col=0)

    # == function calls == #
    ml_df = reshape(df,ml_df)
    ml_df_cleaned = resample(df,ml_df)

    ml_df_cleaned.to_csv(datapath + "all_ml_data_cleaned.csv")

if __name__ == '__main__':
    execute()