#生成eicu数据集的train_listfile.csv �?test_listfile.csv
import os
import pandas as pd    

#得到文件夹train下的所有ts文件名称train_names
train_names = os.listdir('data/in-hospital-mortality/train')
#遍历listfile.csv找出train_names中每一个文件的生存状�?
ls = pd.read_csv('data/in-hospital-mortality/listfile.csv')
train_ls = pd.DataFrame(columns = ["stay", "y_true"]) #create a new dataframe
for name in train_names:
    tmp_ls = ls.loc[ls['stay']==name]
    train_ls = train_ls.append(tmp_ls,ignore_index=True)
#生成train_listfile
train_ls.to_csv('data/in-hospital-mortality/train_listfile.csv',index=False)


#同上 
test_ls = pd.DataFrame(columns=['stay','y_true'])
test_names = os.listdir('data/in-hospital-mortality/test')
for name in test_names:
    tmp_ls = ls.loc[ls['stay']==name]
    test_ls = test_ls.append(tmp_ls,ignore_index=True)
    
test_ls.to_csv('data/in-hospital-mortality/test_listfile.csv',index=False)