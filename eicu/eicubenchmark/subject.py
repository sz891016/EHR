from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import pandas as pd

from eicubenchmark.util import *

def read_stays(subject_path):
    stays = dataframe_from_csv_col(os.path.join(subject_path, 'stays.csv'), index_col=None)
    #stays.INTIME = pd.to_datetime(stays.INTIME)
    #stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    stays.sort_values(by=['INTIME', 'OUTTIME'], inplace=True)
    return stays

    
def read_diagnoses(subject_path):
    return dataframe_from_csv_col(os.path.join(subject_path, 'diagnoses.csv'), index_col=None)
    

def read_events(subject_path, remove_null=True):
    events = dataframe_from_csv_col(os.path.join(subject_path, 'events.csv'), index_col=None)
    if remove_null:
        events = events.ix[events.value.notnull()]
    events.patienthealthsystemstayid = events.patienthealthsystemstayid.fillna(value=-1).astype(int)
    events.patientunitstayid = events.patientunitstayid.fillna(value=-1).astype(int)
    events.valueuom = events.valueuom.fillna('').astype(str)
    return events


def read_variables(variable_path):
    return dataframe_from_csv_col(os.path.join(variable_path, 'variables.csv'), index_col=None)

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

def convert_events_to_timeseries(events,variable_column='name',variables=[]):
    metadata = events[['time', 'patientunitstayid']].sort_values(by=['time', 'patientunitstayid'])\
                    .drop_duplicates(keep='first').set_index('time')
    timeseries = events[['time',variable_column,'value']].sort_values(by=['time',variable_column,'value'],axis=0).drop_duplicates(subset=['time',variable_column],keep='last')
    
    timeseries = timeseries.pivot(index = 'time',columns=variable_column,values='value').merge(metadata,left_index=True,right_index=True).sort_index(axis=0).reset_index()
    
    for v in variables:
        if v not in timeseries:
            timeseries[v] = np.nan
    #为了取值与MIMIC对齐
    timeseries['Glascow coma scale eye opening'] = timeseries['Glascow coma scale eye opening'].map(mapeye)
    timeseries['Glascow coma scale motor response'] = timeseries['Glascow coma scale motor response'].map(mapmotor)
    timeseries['Glascow coma scale total'] = timeseries['Glascow coma scale total'].map(maptotal)
    timeseries['Glascow coma scale verbal response'] = timeseries['Glascow coma scale verbal response'].map(mapverbal)
    return timeseries
    
def get_events_for_stay(events, icustayid, intime=None, outtime=None):
    idx = (events.patientunitstayid == icustayid)
    if outtime is not None:
        idx = idx | ((events.time >= 0) & (events.time <= outtime))
    events = events.ix[idx]
    del events['patientunitstayid']
    return events
    
    
def add_hours_elpased_to_events(events, dt, remove_charttime=True):
    events['HOURS'] = events.time.apply(lambda s: round(s / 60.,4))
    if remove_charttime:
        del events['time']
    return events

    
    
def get_first_valid_from_timeseries(timeseries, variable):
    if variable in timeseries:
        idx = timeseries[variable].notnull()
        if idx.any():
            loc = np.where(idx)[0][0]
            return timeseries[variable].iloc[loc]
    return np.nan