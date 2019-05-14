import os
import pandas as pd

def mapeye(s):
    res = s
    if s == 1.0 or s==1:
        res = '1 No Response'
    if s == 2.0 or s==2:
        res = '2 To pain'
    if s == 3.0 or s==3:
        res = '3 To speech'
    if s == 4.0 or s==4:
        res = '4 Spontaneously'
    return res   
 
def mapmotor(s):
    res = s
    if s == 1.0 or s==1:
        res = "1 No Response"
    if s == 2.0 or s==2:
        res = "2 Abnorm extensn"
    if s == 3.0 or s==3:
        res = "3 Abnorm flexion"
    if s == 4.0 or s==4:
        res = "4 Flex-withdraws"
    if s == 5.0 or s==5:
        res = "5 Localizes Pain"
    if s == 6.0 or s==6:
        res = "6 Obeys Commands"
    return res
    
def maptotal(s):
    res = s
    if s == 3.0 or s==3:
        res = '3'
    if s == 4.0 or s==4:
        res = '4'
    if s == 5.0 or s==5:
        res = '5'
    if s == 6.0 or s==6:
        res = '6'
    if s == 7.0 or s==7:
        res = '7'
    if s == 8.0 or s==8:
        res = "8"
    if s == 9.0 or s==9:
        res = "9"
    if s == 10.0 or s==10:
        res = "10"
    if s == 11.0 or s==11:
        res = "11"
    if s == 12.0 or s==12:
        res = "12"
    if s == 13.0 or s==13:
        res = "13"
    if s == 14.0 or s==14:
        res = "14"
    if s == 15.0 or s==15:
        res = "15"
    if s == "Unable to score due to medication":
        res = ""
    return res       

def mapverbal(s):
    res = s
    if s == 1.0 or s==1:
        res = "1 No Response"
    if s == 2.0 or s==2:
        res = "2 Incomp sounds"
    if s == 3.0 or s==3:
        res = "3 Inapprop words"
    if s == 4.0 or s==4:
        res = "4 Confused"
    if s == 5.0 or s==5:
        res = "5 Oriented"
    return res         

df = pd.read_csv('./data/in-hospital-mortality/val_listfile.csv')
#df = pd.read_csv('./data/length-of-stay/val_listfile.csv')
names = list(df['stay'])
i=1
print('=======>processing val')
for name in names:
    i=i+1
    timeseries = pd.read_csv(os.path.join('./data/in-hospital-mortality/train',name),header = 0)
    #timeseries = pd.read_csv(os.path.join('./data/length-of-stay/train',name),header = 0)
    timeseries['Glascow coma scale eye opening'] = timeseries['Glascow coma scale eye opening'].map(mapeye)
    timeseries['Glascow coma scale motor response'] = timeseries['Glascow coma scale motor response'].map(mapmotor)
    timeseries['Glascow coma scale total'] = timeseries['Glascow coma scale total'].map(maptotal)
    timeseries['Glascow coma scale verbal response'] = timeseries['Glascow coma scale verbal response'].map(mapverbal)
    timeseries.to_csv(os.path.join('./data/in-hospital-mortality/train',name),index=0)
    #timeseries.to_csv(os.path.join('./data/length-of-stay/train',name),index=0)
    if(i%1000==0):
        print(i)
    
df1 = pd.read_csv('./data/in-hospital-mortality/train_listfile.csv')
#df1 = pd.read_csv('./data/length-of-stay/train_listfile.csv')
train_names = list(df1['stay'])
j=1
print('=======>processing train')
for name in train_names:
    j=j+1
    timeseries = pd.read_csv(os.path.join('./data/in-hospital-mortality/train',name),header = 0)
    #timeseries = pd.read_csv(os.path.join('./data/length-of-stay/train',name),header = 0)
    timeseries['Glascow coma scale eye opening'] = timeseries['Glascow coma scale eye opening'].map(mapeye)
    timeseries['Glascow coma scale motor response'] = timeseries['Glascow coma scale motor response'].map(mapmotor)
    timeseries['Glascow coma scale total'] = timeseries['Glascow coma scale total'].map(maptotal)
    timeseries['Glascow coma scale verbal response'] = timeseries['Glascow coma scale verbal response'].map(mapverbal)
    timeseries.to_csv(os.path.join('./data/in-hospital-mortality/train',name),index=0)
    #timeseries.to_csv(os.path.join('./data/length-of-stay/train',name),index=0)
    if j%1000==0:
        print(j)    


df1 = pd.read_csv('./data/in-hospital-mortality/test_listfile.csv')
#df1 = pd.read_csv('./data/length-of-stay/test_listfile.csv')
test_names = list(df1['stay'])
k=1
print('=======>processing test')
for name in test_names:
    k=k+1
    timeseries = pd.read_csv(os.path.join('./data/in-hospital-mortality/test',name),header = 0)
    #timeseries = pd.read_csv(os.path.join('./data/length-of-stay/test',name),header = 0)
    timeseries['Glascow coma scale eye opening'] = timeseries['Glascow coma scale eye opening'].map(mapeye)
    timeseries['Glascow coma scale motor response'] = timeseries['Glascow coma scale motor response'].map(mapmotor)
    timeseries['Glascow coma scale total'] = timeseries['Glascow coma scale total'].map(maptotal)
    timeseries['Glascow coma scale verbal response'] = timeseries['Glascow coma scale verbal response'].map(mapverbal)
    timeseries.to_csv(os.path.join('./data/in-hospital-mortality/test',name),index=0)
    #timeseries.to_csv(os.path.join('./data/length-of-stay/test',name),index=0)
    if k%1000==0:
        print(k)
