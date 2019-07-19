#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:38:43 2019

@author: mohitbeniwal
"""
import pandas as pd
import numpy as np
import traceback
import matplotlib.pyplot as plt 
from collections import Counter

# Compute knn
def knn(df_train, df_test,k):
    try:
        # predicting label
        df_test['pred'] = '-' 
        for i in range (len(df_test)):
            idx=df_test.index[i]
            a = np.array(df_test.iloc[i])
            dist_list = []
            for j in range (len(df_train)):
                b = np.array(df_train.iloc[j])
                d = 0
                for n in range(len(a)-1):
                    if(type(a[n]) is str and type(b[n]) is str):
                        if(a[n] != b[n]):
                             d += 1
                    else:
                        d += pow(a[n] - b[n],2)
                dist = np.sqrt(d)
                dist_list.append(dist)
            # index of k nearest distances
            min_n_idx = np.argsort(dist_list)[:k] 
            neighbours = []
            for s in min_n_idx:
                neighbours.append((df_train.iloc[s]['A16']))
            c = Counter(neighbours)
            df_test.loc[idx,'pred']  = c.most_common(1)[0][0] 
    except Exception as e:
        print(e)
        traceback.print_exc()
    return df_test

# plot 
def plot_k_nearest(x,y,name):
    try:
         plt.title('KNN')
         plt.xlabel('K')
         plt.ylabel(name) 
         plt.plot(x,y)
         fig1 = plt.gcf()
         plt.show()
         plt.draw()
         fig1.savefig(name+'.pdf', dpi=100)
    except Exception as e:
        print('Exception in plot_graph')
        print(e)
        return 0
def main():    
    # Read processed file
    df_train_norm = pd.read_csv("crx.training.processed.csv",index_col=0)
    df_test_norm = pd.read_csv('crx.testing.processed.csv',index_col=0)
    
    x_list = []
    y_list = []
    z_list = []
    
    # compute knn
    a=np.arange(1,18,4)
    columnList=['K']
    for k in a:
        columnList.append(str(k))
        x_list.append(k)
        df_final_test = knn(df_train_norm, df_test_norm,k)
        error_rate = np.sum(1*(df_final_test['A16'] != df_final_test["pred"])) / len(df_final_test)
        accuracy = np.sum(1*(df_final_test['A16'] == df_final_test["pred"])) / len(df_final_test)
        y_list.append(error_rate)
        z_list.append(accuracy)
        print('Accuracy for k = '+str(k)+' is : '+str(accuracy))
        print('Error for k = '+str(k)+' is : '+str(error_rate))
    
    z_list.insert(0,'Accuracy')
    y_list.insert(0,'Error')
    acc_err_list = [z_list]+[y_list]
    acc_err_df = pd.DataFrame(acc_err_list,columns=columnList)
    print('Accuracy_Error Rates DF: \n'+str(acc_err_df))
    acc_err_df.to_csv('Accuracy_Error.csv',encoding='utf-8-sig')
    plot_k_nearest(x_list,y_list[1:],'Error')
    plot_k_nearest(x_list,z_list[1:],'Accuracy')
if __name__=="__main__":
    main()
   
