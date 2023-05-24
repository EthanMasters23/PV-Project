#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Last updated: on Wednesday Apr 12 16:03 2022

@author: Ethan Masters

Purpose: Machine Learning Application

Python Version: Python 3.9.13 (main, Aug 25 2022, 18:29:29) 
"""

import os
import pandas as pd
import numpy as np
import re

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

def MLP_v1(df, ml_df):
    
    nan_indices = list(np.where(df.isna()))[0]
    nan_indices = df.iloc[nan_indices].index 
    df = df.drop(nan_indices)
    ml_df = ml_df.drop(nan_indices)
    
    ml_df['Seconds'] = [(time - time.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() for time in ml_df.index]
    ml_df['Day'] = [d.day for d in ml_df.index]
    ml_df['Month'] = [d.month for d in ml_df.index]
    
    for col in df.columns:
        
        print(f'\n\nFor target variable: {col}\n')
        
        test_ml_df = ml_df.copy()
        
        test_ml_df[col] = df[col].copy()
        
        X = test_ml_df.drop([col], axis = 1).to_numpy()
        y = test_ml_df[col].to_numpy() 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.transform(X_test)

        lasso = Lasso(alpha=0.1)
        lasso.fit(X_train_std, y_train)

        print("Coefficients:", list(zip(lasso.coef_,test_ml_df.drop([col], axis = 1).columns)))
        
        
        if col == 'WindSpeed':
            selected_indices = np.where(lasso.coef_ !=0)[0]
        else:
            selected_indices = np.where((lasso.coef_ > 1) | (lasso.coef_ < -1))[0]
            
        if not selected_indices.any(): 
            (print(f'\nNo features meet current criteria for: {col}'))
            continue
        print("\nSelected indices:", list(test_ml_df.drop([col], axis = 1).columns[selected_indices]))

        X_train_selected = X_train[:, selected_indices]
        X_test_selected = X_test[:, selected_indices]
        mlp = MLPRegressor(hidden_layer_sizes=(15,), activation='tanh', solver='adam')
        mlp.fit(X_train_selected, y_train)

        score = mlp.score(X_test_selected, y_test)
        print("\nTest score:", score)

datapath = re.sub(r'Notebooks|Python Scripts','Support Files/',os.getcwd())

# == Load Irradiance Data == #
target_df = pd.read_csv(datapath + 'Irradiance.csv',index_col=0)
target_df.index = pd.to_datetime(target_df.index)

# == Load Training Data == #
training_df = pd.read_csv(datapath + 'all_ml_data_cleaned.csv',index_col=0)
training_df.index = pd.to_datetime(training_df.index)

# MLP_v1(target_df,training_df)


def MLP_v2(df, ml_df):
    
    nan_indices = list(np.where(df.isna()))[0]
    nan_indices = df.iloc[nan_indices].index 
    df = df.drop(nan_indices)
    ml_df = ml_df.drop(nan_indices)
    
    ml_df['Seconds'] = [(time - time.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() for time in ml_df.index]
    ml_df['Day'] = [d.day for d in ml_df.index]
    
    for col in df.columns:

        if col == 'GlobalIR': continue
        
        print(f'\n\nFor target variable: {col}\n')
        
        test_ml_df = ml_df.copy()
        
        test_ml_df[col] = df[col].copy()
        
        if col == 'DirectIR':
            test_ml_df = test_ml_df[['Direct Shortwave Radiation (W/m²) (sfc)', 'Seconds', 'Day', col]]
        if col == 'DiffuseIR':
            test_ml_df = test_ml_df[['Diffuse Shortwave Radiation (W/m²) (sfc)', 'Seconds', 'Day', col]]
        elif col == 'WindSpeed':
            test_ml_df = test_ml_df[['Wind Speed (km/h) (10 m)', 'Seconds', 'Day', col]]
        elif col == "Temperature":
            test_ml_df = test_ml_df[['Temperature (°C) (2 m elevation corrected)', 'Seconds', 'Day', col]]
            
        X = test_ml_df.drop([col], axis = 1).to_numpy()
        y = test_ml_df[col].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.fit_transform(X_test)

        mlp = MLPRegressor(hidden_layer_sizes=(10,), activation='tanh', solver='adam')
        mlp.fit(X_train_std, y_train)
#         mlp.fit(X_train,y_train)
                
#         print("DataFrame:", test_ml_df)
        print("Features:", list(test_ml_df.drop([col], axis = 1).columns))
        print("Target:", col)
        print("Number of layers:", mlp.n_layers_)
        print("Number of outputs:", mlp.n_outputs_)
        print("Output activation:", mlp.out_activation_)
        print("Coefficients:", mlp.coefs_)
        print("Intercepts (bias vector corresponding to layer i + 1):", mlp.intercepts_)
        
        score = mlp.score(X_test, y_test)
        print("\nTest score:", score)
    
        
MLP_v2(target_df,training_df)