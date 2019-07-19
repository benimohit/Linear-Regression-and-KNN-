#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 21:41:36 2019

@author: mohitbeniwal
"""

import seaborn as sns
import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

def calculate_Plot(df_norm,title):
    df_norm_t = df_norm[['HOM']]
    y = np.array(df_norm_t).squeeze()
    # Iterating to find the possible 3rd attribute
    error_list = []
    column=['UEMP','MAN','LIC', 'GR','NMAN','GOV','HE']
    for i in column:
        phi = 0.0
        t = 0.0
        r = 0.0        
        predictors = df_norm[['FTP','WE',i]]
        X = np.array(predictors)
        
        # Wml formula
        weight = np.linalg.inv((X.T).dot(X)).dot(X.T).dot(y)
        
        # Calculating bias term W0
        for j in range(len(X)):
            phi += X[j]
            t += y[j]
        phi_avg = phi/len(X)
        t_avg = t/len(y)
        for k in range(len(phi)):
            r += weight[k]*phi_avg[k]
        wo = t_avg - r
        # Find loss
        loss = 0.0
        for n in range(len(X)):
            loss += (y[n] - wo - (weight.T).dot(X[n]))**2
        loss /= 2
        error_list.append(loss)
    fig=sns.barplot(x=column, y=error_list)
    fig.set(xlabel='Attributes', ylabel='Loss')
    plt.title(title)
    third_var = column[error_list.index(min(error_list))]   
    print('Third variable '+title+ ' is: '+third_var)
    
# main
def main():
    # get data
    mat_lab = scipy.io.loadmat('detroit.mat')
    data = mat_lab['data']
    columns = ['FTP','UEMP','MAN','LIC', 'GR','NMAN','GOV','HE','WE','HOM']
    df = pd.DataFrame(data,columns=columns)
    #plot without Normalization
    calculate_Plot(df,"Without Normalization")
    plt.show()
    # Normalize the data
    df_new=df.iloc[:,0:9]
    df_norm = (df_new - df_new.mean()) / df_new.std()
    df_norm['HOM']=df.iloc[:,9]
    #plot after Normalization
    calculate_Plot(df_norm,"With Normalization")
    plt.show()

if __name__=="__main__":
    main()



   
    





