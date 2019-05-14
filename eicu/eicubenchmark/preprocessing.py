from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import re

from pandas import DataFrame, Series

from eicubenchmark.util import *

###############################
# Non-time series preprocessing
###############################

g_map = {'Female': 1, 'Male': 2, 'OTHER': 3, '': 0}

def transform_gender(gender_series):
    global g_map
    return { 'Gender': gender_series.fillna('').apply(lambda s: g_map[s] if s in g_map else g_map['OTHER']) }

    
e_map = {'Asian': 1,
         'African American': 2,
         'Native American': 2,
         'Hispanic': 3,
         'Caucasian': 4,
         'UNKNOWN': 0,
         'OTHER': 0,
         '': 0}
 
 
def transform_ethnicity(ethnicity_series):
    global e_map

    def aggregate_ethnicity(ethnicity_str):
        #return ethnicity_str.replace(' OR ', '/').split(' - ')[0].split('/')[0]
        if pd.isnull(ethnicity_str):
            ethnicity_str = ''
        return ethnicity_str.split('/')[0]
       
    ethnicity_series = ethnicity_series.apply(aggregate_ethnicity)
    return {'Ethnicity': ethnicity_series.fillna('').apply(lambda s: e_map[s] if s in e_map else e_map['OTHER'])}

    
def assemble_episodic_data(stays, diagnoses):
    data = {'Icustay': stays.patientunitstayid, 'Age': stays.AGE, 'Length of Stay': stays.LOS,
            'Mortality': stays.MORTALITY}
    data.update(transform_gender(stays.GENDER))
    data.update(transform_ethnicity(stays.ETHNICITY))
    data['Height'] = stays.HEIGHT
    data['Weight'] = stays.WEIGHT
    data = DataFrame(data).set_index('Icustay')
    data = data[['Ethnicity', 'Gender', 'Age', 'Height', 'Weight', 'Length of Stay', 'Mortality']]
    return data.merge(extract_diagnosis_labels(diagnoses), left_index=True, right_index=True)    

    
diagnosis_labels = ['038.9, A41.9','518.81, J96.00','427.31, I48.0','584.9, N17.9','401.9, I10','486, J18.9','491.20, J44.9',                   '780.09, R41.82''428.0, I50.9','518.82','288.8, D72.829','578.9, K92.2','511.9, J91.8','995.9',
                    '799.02, J96.91','276.2, E87.2','799.02, J96.91','427.5, I46.9','244.9, E03.9','585.9, N18.9',
                    '595.9, N30.9','780.6, R50.9','287.5, D69.6','311, F32.9','414.00, I25.10','272.4, E78.5','785.52, R65.21','785.0, R00.0','276.8, E87.6','790.6, R73.9','491.21, J44.1','790.6, R73.9','790.6, R73.9','491.21, J44.1','789.00, R10.9','038.9, R78.81','291.81, F10.239','345.90, R56.9','038.9, R78.81','786.50, R07.9',
                    '428.1, I50.1','276.52, E86.1','518.81, J80','428.1, I50.1','518.81, J80','410.71, I21.4','276.7, E87.5',
                    '300.00, F41.9','285.1, D62','780.57, G47.33','276.1, E87.0, E87.1','786.09, J96.92','507.0, J69.0','585.6, N18.6','250.13, E10.1','434.91, I63.50','432.9, I62.9','415.19, I26.99','995.92, R65.20','436,I67.8','288.9, D72.825','995.92, R65.20','785.51, R57.0','348.30, G93.40','599.0, N39.0',
                    '278.00, E66.9','308.2, F43.0','780.09, R40.0','410.90, I21.3','995.92, R65.2','785.59, R65.21',
                    '162.9, C34.90','573.9, K76.9','401.0, I10','286.9, D68.32','191.9, C71.9','585.3, N18.3','275.41, E83.51',
                    '584.5, N17.0','188.9, C67.9','530.11, K21.0','293.0, F05','493.90, J45','296.80, F31.9','714.0, M06.9','276.1, E87.1','789.5, R18.8','276.51, E86.0','411.89, I24.8','276.0, E87.0','789.5, R18.8',
                    '276.0, E87.0','789.5, R18.8','571.5, K74.60','278.01, E66.01','275.2, E83.42','275.2, E83.42',
                    '799.02, R09.02','780.2, R55','V62.84, R45.851','008.45, A04.7','799.1, R09.2','852.00, S06.6',
                    '424.1, I35.0','008.45, A04.7','411.1, I20.0','560.9, K56.60','441.01, I71.01','682.9, L03.90','185, C61','518.84, J96.00','276.8, E87.8','572.2','263.9, E46','197.0, C78.00','294.9, F03','787.02, R11.0',
                    '345.3, G40.901','427.41, I49.01','S22.4','998.11, I97.89','427.41, I49.01','427.1, I47.2',
                    '518.83, J96.10','414.01, I25.10','860','787.03, R11.10','427.81, R00.1','780.57, G47.30']
'''
,'192.9, C15.9','427.81, R00.1','042, B20','790.4, R74.0','410.11, I21.09','285.9, D64.9','785.59, R57.1','451.9, I80.9','854.06, S06.0','780.01, R40.20','348.31, G93.41','785.59, R57.1, R58','516.3, J84.1',
'''

def extract_diagnosis_labels(diagnoses):
    global diagnosis_labels
    diagnoses['VALUE'] = 1
    labels = diagnoses[['patientunitstayid', 'icd9code', 'VALUE']].drop_duplicates().pivot(index='patientunitstayid', columns='icd9code', values='VALUE').fillna(0).astype(int)
    for l in diagnosis_labels:
        if l not in labels:
            labels[l] = 0
    labels = labels[diagnosis_labels] 
    return labels.rename(dict(zip(diagnosis_labels, ['Diagnosis ' + d for d in diagnosis_labels])), axis=1)
    

def add_hcup_ccs_2015_groups(diagnoses, definitions):
    def_map = {}
    for dx in definitions:
        for code in definitions[dx]['codes']:
            def_map[code] = (dx, definitions[dx]['use_in_benchmark'])
    diagnoses['HCUP_CCS_2015'] = diagnoses.icd9code.apply(lambda c: def_map[c][0] if c in def_map else None)
    diagnoses['USE_IN_BENCHMARK'] = diagnoses.icd9code.apply(lambda c: int(def_map[c][1]) if c in def_map else None)
    return diagnoses 
    

def make_phenotype_label_matrix(phenotypes, stays=None):
    phenotypes = phenotypes[['patientunitstayid', 'HCUP_CCS_2015']].ix[phenotypes.USE_IN_BENCHMARK > 0].drop_duplicates()
    phenotypes['VALUE'] = 1
    phenotypes = phenotypes.pivot(index='patientunitstayid', columns='HCUP_CCS_2015', values='VALUE')
    if stays is not None:
        phenotypes = phenotypes.ix[stays.patientunitstayid.sort_values()]
    return phenotypes.fillna(0).astype(int).sort_index(axis=0).sort_index(axis=1)



def make_variable_csv(eicu_path,output_path):
    vals = dataframe_from_csv(os.path.join(eicu_path, 'apachePredVar.csv'))
    vital = dataframe_from_csv(os.path.join(eicu_path, 'vitalPeriodic.csv'))
    vals['Diastolic blood pressure'] = vital['systemicdiastolic']
    vals['Fraction inspired oxygen'] = vals['fio2']
    vals['Glascow coma scale eye opening'] = vals['eyes']
    vals['Glascow coma scale motor response'] = vals['motor']
    vals['Glascow coma scale total'] = vals['meds']
    vals['Glascow coma scale verbal response'] = vals['verbal']
    vals['Glucose'] = vals['glucose']
    vals['Heart Rate'] = vals['heartrate']
    #vals['Height']
    vals['Mean blood pressure'] = vals['meanbp']
    vals['Oxygen saturation'] = vital['sao2']
    vals['Respiratory rate'] = vals['respiratoryrate']
    vals['Systolic blood pressure'] = vital['systemicsystolic']
    vals['Temperature'] = vals['temperature']
    #vals['Weight']
    vals['pH'] = vals['ph']
    vals = [['Diastolic blood pressure','Fraction inspired oxygen','Glascow coma scale eye opening',
            'Glascow coma scale motor response', 'Glascow coma scale total','Glascow coma scale verbal response', 'Glucose', 'Heart Rate', 'Height','Mean blood pressure', 'Oxygen saturation', 'pH', 'Respiratory rate',
            'Systolic blood pressure', 'Temperature', 'Weight']]
    vals.to_csv(os.path.join(output_path,'variables.csv'),index = False)
    return vals    



###################################
# Time series preprocessing       #
###################################
# SBP: some are strings of type SBP/DBP
def clean_sbp(df):
    v = df.value.astype(str)
    idx = v.apply(lambda s: '/' in s)
    v.ix[idx] = v[idx].apply(lambda s: re.match('^(\d+)/(\d+)$', s).group(1))
    return v.astype(float)


def clean_dbp(df):
    v = df.value.astype(str)
    idx = v.apply(lambda s: '/' in s)
    v.ix[idx] = v[idx].apply(lambda s: re.match('^(\d+)/(\d+)$', s).group(2))
    return v.astype(float)


#CRR: strings with brisk, <3 normal, delayed, or >3 abnormal
def clean_crr(df):
    v = Series(np.zeros(df.shape[0]), index=df.index)
    v[:] = np.nan

    # when df.VALUE is empty, dtype can be float and comparision with string
    # raises an exception, to fix this we change dtype to str
    df.VALUE = df.VALUE.astype(str)

    v.ix[(df.VALUE == 'Normal <3 secs') | (df.VALUE == 'Brisk')] = 0
    v.ix[(df.VALUE == 'Abnormal >3 secs') | (df.VALUE == 'Delayed')] = 1
    return v


# FIO2: many 0s, some 0<x<0.2 or 1<x<20
def clean_fio2(df):
    v = df.VALUE.astype(float)

    ''' The line below is the correct way of doing the cleaning, since we will not compare 'str' to 'float'.
    If we use that line it will create mismatches from the data of the paper in ~50 ICU stays.
    The next releases of the benchmark should use this line.
    '''
    # idx = df.VALUEUOM.fillna('').apply(lambda s: 'torr' not in s.lower()) & (v>1.0)

    ''' The line below was used to create the benchmark dataset that the paper used. Note this line will not work
    in python 3, since it may try to compare 'str' to 'float'.
    '''
    # idx = df.VALUEUOM.fillna('').apply(lambda s: 'torr' not in s.lower()) & (df.VALUE > 1.0)

    ''' The two following lines implement the code that was used to create the benchmark dataset that the paper used.
    This works with both python 2 and python 3.
    '''
    is_str = np.array(map(lambda x: type(x) == str, list(df.VALUE)), dtype=np.bool)
    idx = df.VALUEUOM.fillna('').apply(lambda s: 'torr' not in s.lower()) & (is_str | (~is_str & (v > 1.0)))

    v.ix[idx] = v[idx] / 100.
    return v


# GLUCOSE, PH: sometimes have ERROR as value
def clean_lab(df):
    v = df.value
    idx = v.apply(lambda s: type(s) is str and not re.match('^(\d+(\.\d*)?|\.\d+)$', s))
    v.ix[idx] = np.nan
    return v.astype(float)


# O2SAT: small number of 0<x<=1 that should be mapped to 0-100 scale
def clean_o2sat(df):
    # change "ERROR" to NaN
    v = df.value
    idx = v.apply(lambda s: type(s) is str and not re.match('^(\d+(\.\d*)?|\.\d+)$', s))
    v.ix[idx] = np.nan

    v = v.astype(float)
    idx = (v <= 1)
    v.ix[idx] = v[idx] * 100.
    return v


# Temperature: map Farenheit to Celsius, some ambiguous 50<x<80
def clean_temperature(df):
    v = df.value.astype(float)
    idx = df.valueuom.fillna('').apply(lambda s: 'F' in s.lower()) | (v >= 79)
    v.ix[idx] = (v[idx] - 32) * 5. / 9
    return v


# Weight: some really light/heavy adults: <50 lb, >450 lb, ambiguous oz/lb
# Children are tough for height, weight
def clean_weight(df):
    v = df.VALUE.astype(float)
    # ounces
    idx = df.VALUEUOM.fillna('').apply(lambda s: 'oz' in s.lower()) | df.MIMIC_LABEL.apply(lambda s: 'oz' in s.lower())
    v.ix[idx] = v[idx] / 16.
    # pounds
    idx = idx | df.VALUEUOM.fillna('').apply(lambda s: 'lb' in s.lower()) | df.MIMIC_LABEL.apply(lambda s: 'lb' in s.lower())
    v.ix[idx] = v[idx] * 0.453592
    return v


# Height: some really short/tall adults: <2 ft, >7 ft)
# Children are tough for height, weight
def clean_height(df):
    v = df.VALUE.astype(float)
    idx = df.VALUEUOM.fillna('').apply(lambda s: 'in' in s.lower()) | df.MIMIC_LABEL.apply(lambda s: 'in' in s.lower())
    v.ix[idx] = np.round(v[idx] * 2.54)
    return v


# ETCO2: haven't found yet
# Urine output: ambiguous units (raw ccs, ccs/kg/hr, 24-hr, etc.)
# Tidal volume: tried to substitute for ETCO2 but units are ambiguous
# Glascow coma scale eye opening
# Glascow coma scale motor response
# Glascow coma scale total
# Glascow coma scale verbal response
# Heart Rate
# Respiratory rate
# Mean blood pressure
#episode{}_timeseries中的�?= clean_fns + ['Hours','Glascow...(4)','Heart Rate','Mean blood pressure','Respiratory rate']
clean_fns = {
    #'Capillary refill rate': clean_crr,
    'Diastolic blood pressure': clean_dbp,
    'Systolic blood pressure': clean_sbp,
    #'Fraction inspired oxygen': clean_fio2,
    'Oxygen saturation': clean_o2sat,
    'Glucose': clean_lab,
    #'pH': clean_lab,
    'Temperature': clean_temperature,
    #'Weight': clean_weight,
    #'Height': clean_height
}


def clean_events(events):
    global clean_fns
    for var_name, clean_fn in clean_fns.items():
        idx = (events.name == var_name)
        try:
            #对idx行的'VALUE'列执行相关的清洗函数
            events.ix[idx, 'value'] = clean_fn(events.ix[idx])
        except Exception as e:
            print("Exception in clean_events:", clean_fn.__name__, e)
            print("number of rows:", np.sum(idx))
            print("values:", events.ix[idx])
            exit()
    return events.ix[events.value.notnull()]