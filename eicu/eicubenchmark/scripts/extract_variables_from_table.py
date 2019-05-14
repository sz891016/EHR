from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import numpy as np

from eicubenchmark.util import *

parser = argparse.ArgumentParser(description='Extract all variables from apacheApsVar and VatalPriodic csv files.')
parser.add_argument('eicu_path', type=str, default = '../eicu_demo', help='Directory containing EICU CSV files.')
parser.add_argument('output_path', type=str, default = './eicubenchmark/resources/', help='Directory output variables.csv files.')
args, _ = parser.parse_known_args()

def make_variable_csv(eicu_path,output_path):
    vals = dataframe_from_csv(os.path.join(eicu_path, 'apacheApsVar.csv'))
    vital = dataframe_from_csv(os.path.join(eicu_path, 'vitalPeriodic.csv'))
    vital = vital[['patientunitstayid','systemicdiastolic','systemicsystolic','sao2']]
    vital.rename(columns={'systemicdiastolic':'Diastolic blood pressure',
                           'systemicsystolic':'Systolic blood pressure','sao2':'Oxygen saturation'},inplace = True)
    vals['Fraction inspired oxygen'] = vals['fio2']
    vals['Glascow coma scale eye opening'] = vals['eyes']
    vals['Glascow coma scale motor response'] = vals['motor']
    #vals['Glascow coma scale total'] = vals['meds']
    vals['Glascow coma scale verbal response'] = vals['verbal']
    vals['Glucose'] = vals['glucose']
    vals['Heart Rate'] = vals['heartrate']
    vals['Height'] = np.nan
    vals['Mean blood pressure'] = vals['meanbp']
    #vals['Oxygen saturation'] = vital['sao2'] 
    vals['Respiratory rate'] = vals['respiratoryrate']
    #vals['Systolic blood pressure'] = vital['systemicsystolic']
    vals['Temperature'] = vals['temperature']
    vals['Weight'] = np.nan
    vals['pH'] = vals['ph']
    vals = vals[['patientunitstayid','Fraction inspired oxygen','Glascow coma scale eye opening',
            'Glascow coma scale motor response', 'Glascow coma scale verbal response', 'Glucose', 'Heart Rate', 'Mean blood pressure',  'pH', 'Respiratory rate',
            'Temperature']]
    vals = vals.merge(vital,on = 'patientunitstayid')
    vals.to_csv(os.path.join(output_path,'variables.csv'),index = False)
    print('make_variable_csv done!')
    #return vals
    
if __name__ == '__main__': 
    make_variable_csv(args.eicu_path,args.output_path)