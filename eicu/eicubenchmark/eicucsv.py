from __future__ import absolute_import
from __future__ import print_function

import csv
import numpy as np
import os
import pandas as pd
import sys

from eicubenchmark.util import *


def to_minute(datetime):
    datetime = str(datetime)
    H = int(datetime[0:2])
    M = int(datetime[3:5])
    minute = H*60+M
    return minute


def diagnosispriority2seq_num(priority):
    if priority == 'Primary':
        res = 2
    elif priority == 'Major':
        res = 1
    else:
        res = 0
    return res


def transdischargestatus(status):
    if status == 'Expired':
        res = 1
    elif status == 'Alive':
        res = 0
    else:
        res = 2
    return res

def transage(age):
    age = str(age)
    if age.find('>') != -1:
        return 90
    else:
        return float(age)
        

# def deal_intakeoutput_table(eicu_path):
    # intakeoutput = dataframe_from_csv(os.path.join(eicu_path, 'intakeOutput.csv'))
    # intake = intakeoutput[['patientunitstayid','intaketotal','intakeoutputoffset']]
    # intake['valueuom'] = intakeoutput.celllabel.apply(lambda x:x[x.find('(')+1:x.find(')')] if x is not None else None)
    # intake['time'] = intake['intakeoutputoffset']
    # intake['']
        
def deal_infusiondrug_table(eicu_path):
    ids = read_ids_from_patient(eicu_path)
    drug = dataframe_from_csv(os.path.join(eicu_path, 'infusionDrug.csv'))
    drug = drug.merge(ids)
    drug['time'] = drug['infusionoffset']
    drug['name'] = drug['drugname'].apply(lambda x: x[0:x.find('(')-1] if x is not None else None)
    drug['value'] = drug['drugrate']
    drug['valueuom'] = drug['drugname'].apply(lambda x:x[x.find('(')+1:x.find(')')] if x is not None else None )
    drug = drug[['uniquepid','patienthealthsystemstayid','patientunitstayid','time','name','value','valueuom']]
    drug.to_csv(os.path.join(eicu_path, 'infusionDrug_deal.csv'), index=False)
    print('infusionDrug_deal.csv write done!')
    
    
def deal_lab_table(eicu_path):
    ids = read_ids_from_patient(eicu_path)
    lab = dataframe_from_csv(os.path.join(eicu_path, 'lab.csv'))
    lab = lab.merge(ids)
    lab['time'] = lab['labresultoffset']
    lab['name'] = lab['labname']
    lab['value'] = lab['labresult']
    lab['valueuom'] = lab['labmeasurenamesystem']
    lab = lab[['uniquepid','patienthealthsystemstayid','patientunitstayid','time','name','value','valueuom']]
    lab.to_csv(os.path.join(eicu_path, 'lab_deal.csv'),index = False)
    print('lab_deal.csv write done!')

        
    
def read_ids_from_patient(eicu_path):
    pats = dataframe_from_csv(os.path.join(eicu_path, 'patient.csv'))
    ids = pats[['uniquepid','patienthealthsystemstayid','patientunitstayid']]
    ids['uniquepid'] = ids['uniquepid'].apply(lambda x :x[0:3]+ x[4:])
    ids['uniquepid'] = ids['uniquepid'].apply(lambda x:int(x))
    #ids.to_csv(os.path.join(eicu_path, 'all_ids.csv'), index=False)
    return ids
 
 
def read_patients_table(eicu_path):
    pats = dataframe_from_csv(os.path.join(eicu_path, 'patient.csv'))
    ids = read_ids_from_patient(eicu_path)
    pats['uniquepid'] = ids['uniquepid']
    pats = pats[['uniquepid', 'gender']]
    return pats


def read_admissions_table(eicu_path):
    admits = dataframe_from_csv(os.path.join(eicu_path, 'patient.csv'))
    # admits['tmptime'] = admits['unitadmittime24'].map(to_minute)
    # admits['admittime'] = admits.apply(lambda x: x['tmptime'] + x['hospitaladmitoffset'], axis=1)
    # admits['dischtime'] = admits.apply(lambda x: x['tmptime'] + x['hospitaldischargeoffset'],axis = 1)
    admits['admittime'] = admits['hospitaladmitoffset']
    admits['dischtime'] = admits['hospitaldischargeoffset']
    ids = read_ids_from_patient(eicu_path)
    admits['uniquepid'] = ids['uniquepid']
    admits = admits[['uniquepid', 'patienthealthsystemstayid','admittime','dischtime', 'ethnicity', 'apacheadmissiondx']]
    return admits 


def read_icustays_table(eicu_path):
    stays = dataframe_from_csv(os.path.join(eicu_path, 'patient.csv'))
    ids = read_ids_from_patient(eicu_path)
    stays['uniquepid'] = ids['uniquepid']
    stays['INTIME'] = stays['unitadmittime24'].map(to_minute)
    # stays['OUTTIME'] = stays.apply(lambda x: x['INTIME'] + x['unitdischargeoffset'],axis = 1)
    stays['OUTTIME'] = stays['unitadmittime24'].map(to_minute)+stays['unitdischargeoffset']

    stays['AGE'] = stays['age'].map(transage)
    stays['LOS'] =stays.apply(lambda x: round(((x['OUTTIME'])/60./24.),4),axis = 1)
    stays['GENDER'] = stays['gender']
    stays['ETHNICITY'] = stays['ethnicity']
    stays['HEIGHT'] = stays['admissionheight']
    stays['WEIGHT'] = stays['admissionweight']
    stays = stays[['uniquepid','patienthealthsystemstayid','patientunitstayid','hospitaldischargestatus','unitdischargestatus','INTIME','OUTTIME','AGE','LOS','GENDER','ETHNICITY','HEIGHT','WEIGHT']]
    return stays

def read_icd_diagnoses_table(eicu_path):
    ids = read_ids_from_patient(eicu_path)
    diagnoses = dataframe_from_csv(os.path.join(eicu_path, 'diagnosis.csv'))
    diagnoses = diagnoses[['patientunitstayid','icd9code', 'diagnosisstring','diagnosispriority']]
    diagnoses['seq_num'] = diagnoses['diagnosispriority'].map(diagnosispriority2seq_num)
    del diagnoses['diagnosispriority']
    diagnoses = diagnoses.merge(ids, how='inner', left_on='patientunitstayid', right_on='patientunitstayid')
    diagnoses[['uniquepid','patienthealthsystemstayid', 'seq_num']] = diagnoses[['uniquepid', 'patienthealthsystemstayid', 'seq_num']].astype(int)
    diagnoses['icd9code'] = diagnoses['icd9code'].fillna('999')
    indexs = diagnoses[(diagnoses.icd9code=='999')].index.tolist()
    diagnoses = diagnoses.drop(indexs)
    return diagnoses
 

def rename_nursechartcelltypevalname(name):
    if name == 'Invasive BP Diastolic' or name == 'Non-Invasive BP Diastolic':
        res =  'Diastolic blood pressure'
    elif name == 'Eyes':
        res = 'Glascow coma scale eye opening'
    elif name == 'Motor':
        res = 'Glascow coma scale motor response'
    elif name == 'GCS Total':
        res = 'Glascow coma scale total'
    elif name == 'Verbal':
        res = 'Glascow coma scale verbal response'
    elif name == 'Bedside Glucose':
        res = 'Glucose'
    elif name == 'Non-Invasive BP Mean':
        res = 'Mean blood pressure'
    elif name == 'O2 Saturation':
        res = 'Oxygen saturation'
    elif name == 'Respiratory Rate':
        res = 'Respiratory rate'
    elif name == 'Invasive BP Systolic' or name == 'Non-Invasive BP Systolic':
        res = 'Systolic blood pressure'
    elif name == 'Temperature (C)' or name == 'Temperature (F)':
        res = 'Temperature'
    elif name == 'Heart Rate':
      res = name
    return res
 
 
def read_event_from_nursecharting(eicu_path):
    nursechart = dataframe_from_csv(os.path.join(eicu_path, 'nurseCharting.csv'))
    nursechart = nursechart[['patientunitstayid','nursingchartoffset','nursingchartcelltypevalname','nursingchartvalue']]
    df1 = nursechart.loc[nursechart.nursingchartcelltypevalname=='Invasive BP Diastolic'] 
    df2 = nursechart.loc[nursechart.nursingchartcelltypevalname=='Non-Invasive BP Diastolic'] 
    df3 = nursechart.loc[nursechart.nursingchartcelltypevalname=='Eyes'] 
    df4 = nursechart.loc[nursechart.nursingchartcelltypevalname=='Motor'] 
    df5 = nursechart.loc[nursechart.nursingchartcelltypevalname=='GCS Total'] 
    df6 = nursechart.loc[nursechart.nursingchartcelltypevalname=='Verbal'] 
    df7 = nursechart.loc[nursechart.nursingchartcelltypevalname=='Bedside Glucose'] 
    df8 = nursechart.loc[nursechart.nursingchartcelltypevalname=='Heart Rate'] 
    df9 = nursechart.loc[nursechart.nursingchartcelltypevalname=='Non-Invasive BP Mean'] 
    df10 = nursechart.loc[nursechart.nursingchartcelltypevalname=='O2 Saturation']
    df11 = nursechart.loc[nursechart.nursingchartcelltypevalname=='Respiratory Rate'] 
    df12 = nursechart.loc[nursechart.nursingchartcelltypevalname=='Invasive BP Systolic']
    df13 = nursechart.loc[nursechart.nursingchartcelltypevalname=='Non-Invasive BP Systolic']
    df14 = nursechart.loc[nursechart.nursingchartcelltypevalname=='Temperature (C)']
    df15 = nursechart.loc[nursechart.nursingchartcelltypevalname=='Temperature (F)']
    nursechart = df1.append(df2).append(df3).append(df4).append(df5).append(df6).append(df7).append(df8).append(df9).append(df10).append(df11).append(df12).append(df13).append(df14).append(df15)
    ids = read_ids_from_patient(eicu_path)
    nursechart = ids.merge(nursechart, how='inner', left_on='patientunitstayid', right_on='patientunitstayid')
    nursechart['name'] = nursechart.nursingchartcelltypevalname.map(rename_nursechartcelltypevalname)
    del nursechart['nursingchartcelltypevalname']
    nursechart['valueuom'] = np.nan
    nursechart.rename(columns={'nursingchartoffset':'time','nursingchartvalue':'value'},inplace = True)
    return nursechart

# def read_events_table_by_row(eicu_path, table):
    # nb_rows = {'lab_deal': 278110, 'infusiondrug_deal': 22975}
    # if table == 'lab':
        # deal_lab_table(eicu_path)
    # if table == 'infusionDrug':
        # deal_infusiondrug_table(eicu_path)
    # reader = csv.DictReader(open(os.path.join(eicu_path, table + '_deal.csv'), 'r'))
    # for i, row in enumerate(reader):
        # if 'patientunitstayid' not in row:
            # row['patientunitstayid'] = ''
        # yield row, i, nb_rows[(table+'_deal').lower()]


def merge_on_subject_admission(table1, table2):
    return table1.merge(table2, how='inner', left_on=['uniquepid', 'patienthealthsystemstayid'], right_on=['uniquepid', 'patienthealthsystemstayid'])        


def merge_on_subject(table1, table2):
    return table1.merge(table2, how='inner', left_on=['uniquepid'], right_on=['uniquepid'])

        
def filter_admissions_on_nb_icustays(stays, min_nb_stays=1, max_nb_stays=1):
    to_keep = stays.groupby('patienthealthsystemstayid').count()[['patientunitstayid']].reset_index()
    to_keep = to_keep.ix[(to_keep.patientunitstayid >= min_nb_stays) & (to_keep.patientunitstayid <= max_nb_stays)][['patienthealthsystemstayid']]
    stays = stays.merge(to_keep, how='inner', left_on='patienthealthsystemstayid', right_on='patienthealthsystemstayid')
    return stays

   
def add_inunit_mortality_to_icustays(stays):
    stays['MORTALITY_INUNIT'] = stays['unitdischargestatus'].map(transdischargestatus)
    del stays['unitdischargestatus']
    return stays
    
def add_inhospital_mortality_to_icustays(stays):
    stays['MORTALITY'] = stays['hospitaldischargestatus'].map(transdischargestatus)
    stays['MORTALITY_INHOSPITAL'] = stays['MORTALITY']
    del stays['hospitaldischargestatus']
    return stays



def filter_icustays_on_mortality(stays):
    stays = stays.ix[(stays.MORTALITY!=2)|(stays.MORTALITY_INUNIT!=2)]
    return stays

    
def filter_icustays_on_age(stays, min_age=18, max_age=np.inf):
    stays = stays.ix[(stays.AGE >= min_age) & (stays.AGE <= max_age)]
    return stays

def filter_diagnoses_on_stays(diagnoses, stays):
    return diagnoses.merge(stays[['uniquepid', 'patienthealthsystemstayid', 'patientunitstayid']].drop_duplicates(), how='inner',
                           left_on=['uniquepid', 'patienthealthsystemstayid','patientunitstayid'], right_on=['uniquepid', 'patienthealthsystemstayid','patientunitstayid'])

def filter_events_on_stays(events,stays):
    return events.merge(stays[['uniquepid', 'patienthealthsystemstayid', 'patientunitstayid']].drop_duplicates(), how='inner',
                           left_on=['uniquepid', 'patienthealthsystemstayid','patientunitstayid'], right_on=['uniquepid', 'patienthealthsystemstayid','patientunitstayid'])
    
def count_icd_codes(diagnoses, output_path=None):
    codes = diagnoses[['icd9code', 'diagnosisstring']].drop_duplicates().set_index('icd9code')
    codes['COUNT'] = diagnoses.groupby('icd9code')['patientunitstayid'].count()
    codes.COUNT = codes.COUNT.fillna(0).astype(int)
    codes = codes.ix[codes.COUNT > 0]
    if output_path:
        codes.to_csv(output_path, index_label='icd9code')
    # codes.rename(columns={'icd9code':'ICD9_CODE'},inplace=True)
    return codes.sort_values(by='COUNT', ascending=False).reset_index()

def break_up_stays_by_subject(stays, output_path, subjects=None, verbose=1):
    subjects = stays.uniquepid.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for i, subject_id in enumerate(subjects):
        if verbose:
            sys.stdout.write('\rSUBJECT {0} of {1}...'.format(i+1, nb_subjects))
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass
        
        stays.ix[stays.uniquepid == subject_id].sort_values(by='INTIME').to_csv(os.path.join(dn, 'stays.csv'), index=False)
    if verbose:
        sys.stdout.write('DONE!\n')

    
def break_up_diagnoses_by_subject(diagnoses, output_path, subjects=None, verbose=1):
    subjects = diagnoses.uniquepid.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for i, subject_id in enumerate(subjects):
        if verbose:
            sys.stdout.write('\rSUBJECT {0} of {1}...'.format(i+1, nb_subjects))
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        diagnoses.ix[diagnoses.uniquepid == subject_id].sort_values(by=['patientunitstayid', 'seq_num']).to_csv(os.path.join(dn, 'diagnoses.csv'), index=False)
    if verbose:
        sys.stdout.write('DONE!\n')

        
def break_up_events_by_subject(events, output_path, subjects=None, verbose=1):        
    subjects = events.uniquepid.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for i,subject_id in enumerate(subjects):
        if verbose:
            sys.stdout.write('\rSUBJECT {0} of {1}...'.format(i+1, nb_subjects))
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass
        events.ix[events.uniquepid==subject_id].sort_values(by="patientunitstayid").to_csv(os.path.join(dn, 'events.csv'), index=False)
    if verbose:
        sys.stdout.write('DONE!\n')
        

        