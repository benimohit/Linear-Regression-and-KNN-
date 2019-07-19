#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 12:47:52 2019

@author: mohitbeniwal
"""
import pandas as pd
import numpy as np
import traceback

# method to get dataFrame
def get_df(data):
    df=[]
    for i in data:
        i=i[:-1]
        j=i.split(',')
        df.append(j)
    df=pd.DataFrame(df,columns=['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15','A16'])
    #replace "?" with NaN
    df.replace('?', np.NaN,inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')
    return df

# this method replaces non-numeric columns with mode value
def replace_with_mode(df):
    try:
        # get categorical columns
        cat_c = df.columns[df.dtypes.apply(lambda c: np.issubdtype(c, np.object_))]
        for col in cat_c:
                df[col].fillna(df[col].mode()[0], inplace=True)
    except Exception as e:
        print(e)
        traceback.print_exc()
    return df

# Function to normalize data
def normalize_colmns(df):
    try:
        # columns with numerical data
        num_c = df._get_numeric_data().columns
        for col in num_c:
            #col_zscore = col + '_zscore'
            df[col] = (df[col] - df[col].mean())/df[col].std()
    except Exception as e:
        print(e)
        traceback.print_exc()
    return df

def main():
    # Read files
    crx_train= open('crx.data.training', 'r')
    crx_data_train=crx_train.readlines()
    
    crx_test= open('crx.data.testing', 'r')
    crx_data_test=crx_test.readlines()
    
    # train data
    df_train=get_df(crx_data_train)
    # test data
    df_test=get_df(crx_data_test)
    
    df_train=replace_with_mode(df_train)
    df_test=replace_with_mode(df_test)
    
    # separate +ve and _ve data
    df_pos_train=df_train.loc[df_train.iloc[:,15]=='+'].reset_index(drop=True)
    df_neg_train=df_train.loc[df_train.iloc[:,15]=='-'].reset_index(drop=True)
    df_pos_test=df_test.loc[df_test.iloc[:,15]=='+'].reset_index(drop=True)
    df_neg_test=df_test.loc[df_test.iloc[:,15]=='-'].reset_index(drop=True)
     # replace numerical columns with column mean
    df_pos_train.fillna(df_pos_train.mean(),inplace=True)
    df_neg_train.fillna(df_neg_train.mean(),inplace=True) 
    df_pos_test.fillna(df_pos_train.mean(),inplace=True) 
    df_neg_test.fillna(df_neg_train.mean(),inplace=True) 
    
    # Normalize the data
    df_train_norm = normalize_colmns(df_pos_train.append(df_neg_train))
    df_train_norm.index = range(len(df_train_norm.index)) # resetting index
    
    df_test_norm = normalize_colmns(df_pos_test.append(df_neg_test))
    df_test_norm.index = range(len(df_test_norm.index))
    
    df_train_norm.to_csv('crx.training.processed.csv',encoding='utf-8-sig')
    df_test_norm.to_csv('crx.testing.processed.csv',encoding='utf-8-sig')

if __name__=="__main__":
    main()