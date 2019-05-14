from __future__ import absolute_import
from __future__ import print_function

import argparse
import yaml
import os

from eicubenchmark.eicucsv import *
#from eicubenchmark.preprocessing import add_hcup_ccs_2015_groups, make_phenotype_label_matrix
from eicubenchmark.util import *


parser = argparse.ArgumentParser(description='Extract per-subject data from EICU CSV files.')
parser.add_argument('eicu_path', type=str, help='Directory containing EICU CSV files.')
parser.add_argument('output_path', type=str, help='Directory where per-subject data should be written.')
parser.add_argument('--event_tables', '-e', type=str, nargs='+', help='Tables from which to read events.',
                    default=['nursecharting'])
parser.add_argument('--phenotype_definitions', '-p', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/hcup_ccs_2015_definitions.yaml'),
                    help='YAML file with phenotype definitions.')
parser.add_argument('--itemids_file', '-i', type=str, help='CSV containing list of ITEMIDs to keep.')
parser.add_argument('--verbose', '-v', type=int, help='Level of verbosity in output.', default=1)
parser.add_argument('--test', action='store_true', help='TEST MODE: process only 1000 subjects, 1000000 events.')
args, _ = parser.parse_known_args()

try:
    os.makedirs(args.output_path)
except:
    pass
    

patients = read_patients_table(args.eicu_path)
admits = read_admissions_table(args.eicu_path)
stays = read_icustays_table(args.eicu_path)
if args.verbose:
    print('START:', stays.patientunitstayid.unique().shape[0], stays.patienthealthsystemstayid.unique().shape[0],
          stays.uniquepid.unique().shape[0])

stays = merge_on_subject_admission(stays, admits)
stays = merge_on_subject(stays, patients)    
stays = filter_admissions_on_nb_icustays(stays)    
if args.verbose:
    print('REMOVE MULTIPLE STAYS PER ADMIT:', stays.patientunitstayid.unique().shape[0], stays.patienthealthsystemstayid.unique().shape[0],
          stays.uniquepid.unique().shape[0]) 
          
stays = add_inunit_mortality_to_icustays(stays)
stays = add_inhospital_mortality_to_icustays(stays)
stays = filter_icustays_on_mortality(stays)
stays = filter_icustays_on_age(stays)
if args.verbose:
    print('REMOVE PATIENTS AGE < 18:', stays.patientunitstayid.unique().shape[0], stays.patienthealthsystemstayid.unique().shape[0],
          stays.uniquepid.unique().shape[0])

# stays.rename(columns={'uniquepid':'SUBJECT_ID','patienthealthsystemstayid':'HADM_ID','patientunitstayid':'ICUSTAY_ID'},inplace=True)
stays.to_csv(os.path.join(args.output_path, 'all_stays.csv'), index=False)
diagnoses = read_icd_diagnoses_table(args.eicu_path)
diagnoses = filter_diagnoses_on_stays(diagnoses, stays)
diagnoses.to_csv(os.path.join(args.output_path, 'all_diagnoses.csv'), index=False)
count_icd_codes(diagnoses, output_path=os.path.join(args.output_path, 'diagnosis_counts.csv'))
events = read_event_from_nursecharting(args.eicu_path)
events = filter_events_on_stays(events,stays)
events.to_csv(os.path.join(args.output_path, 'all_events.csv'), index=False)




subjects = stays.uniquepid.unique()
break_up_stays_by_subject(stays, args.output_path, subjects=subjects, verbose=args.verbose)
break_up_diagnoses_by_subject(diagnoses, args.output_path, subjects=subjects, verbose=args.verbose)  
break_up_events_by_subject(events,args.output_path, subjects=subjects, verbose=args.verbose)



                                              