import os
import pandas as pd  

ls = pd.read_csv('data/in-hospital-mortality/listfile.csv')
train_ls = pd.read_csv('data/in-hospital-mortality/train_listfile.csv')
test_ls = pd.read_csv('data/in-hospital-mortality/test_listfile.csv')

ls = ls[~ls['y_true'].isin([2])]
train_ls = train_ls[~train_ls['y_true'].isin([2])]
test_ls = test_ls[~test_ls['y_true'].isin([2])]

ls.to_csv('data/in-hospital-mortality/listfile.csv',index=False)
train_ls.to_csv('data/in-hospital-mortality/train_listfile.csv',index=False)
test_ls.to_csv('data/in-hospital-mortality/test_listfile.csv',index=False)