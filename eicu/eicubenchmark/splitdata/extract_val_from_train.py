import os
import pandas as pd

train_df = pd.read_csv('data/in-hospital-mortality/train_listfile.csv',header=0)
val_df = train_df.sample(frac=0.15,axis=0)
val_df.to_csv('data/in-hospital-mortality/val_listfile.csv',index =False )

#delete recoder in train_listfile which contained in val_listfile
val_df = pd.read_csv('data/in-hospital-mortality/val_listfile.csv')
val_names = list(val_df['stay'])

train_df = train_df[~train_df['stay'].isin(val_names)]
train_df.to_csv('data/in-hospital-mortality/train_listfile.csv',index = False)
    

