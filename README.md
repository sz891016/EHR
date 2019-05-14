# EHR

mimic3benchmark文件夹：处理MIMIC-III数据集的代码
mimic3models：读取mimic数据进行相关任务预测的代码
mimic3models/keras_models：包含所有的模型代码

eicu
eicubenchmark文件夹：处理EICU数据集的代码
mimic3models：读取eicu数据进行相关任务预测，
              其中combin_in_hospital_mortality（混合数据集），in_hospital_mortality，length_of_stay，logistic是利用eicu数据进行对应的任务预测
              其余代码与EHR/mimic3models中代码相同
