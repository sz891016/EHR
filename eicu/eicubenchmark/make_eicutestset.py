import os
import random
import pandas as pd

df = pd.read_csv('./eicubenchmark/resources/patient_status.csv',header=0)
df = df[~df['MORTALITY'].isin([2])]

df0 = df.loc[df['MORTALITY']==0]
df1 = df.loc[df['MORTALITY']==1]
li0 = list(df0['uniquepid'])

print('======> drop some patient_status_0')
rows = []
rows = random.sample(li0,24782)
print(len(rows))
df = df[~df['uniquepid'].isin(rows)]
print('regenerater patient_status')
df.to_csv('./eicubenchmark/resources/patient_status.csv',index=False)




df = pd.read_csv('./eicubenchmark/resources/patient_status.csv',header=0)
df0 = df.loc[df['MORTALITY']==0]
df1 = df.loc[df['MORTALITY']==1]
li0 = list(df0['uniquepid'])
li1 = list(df1['uniquepid'])
l0 = random.sample(li0,54547)
l1 = random.sample(li1,8513)
trainlis = l0+l1

def func(a): 
    if a in trainlis:
        return 0
    else:
        return 1

print('labeling train and test')
df['flag'] = df['uniquepid'].apply(lambda x:func(x))
print('finished labeling train and test')
del df['MORTALITY']
df.to_csv('./eicubenchmark/resources/testset.csv',index=False)     