from __future__ import absolute_import
from __future__ import print_function

import os
import shutil
import argparse
import pandas as pd

def is_subject_folder(x):
    return str.isdigit(x)
    
    
    
def main():

    n_events = 0                   
    empty_hadm = 0                
    no_hadm_in_stay = 0            
    no_icustay = 0                 
    recovered = 0                  
    could_not_recover = 0          
    icustay_missing_in_stays = 0  

    parser = argparse.ArgumentParser()
    parser.add_argument('subjects_root_path', type=str,
                        help='Directory containing subject subdirectories.')
    args = parser.parse_args()
    print(args)
    
    subdirectories = os.listdir(args.subjects_root_path)
    subjects = list(filter(is_subject_folder, subdirectories))   
    
    for (index, subject) in enumerate(subjects):
        if index % 100 == 0:
            print("processed {} / {} {}\r".format(index+1, len(subjects), ' '*10))
        
        stays_df = pd.read_csv(os.path.join(args.subjects_root_path, subject, 'stays.csv'), index_col=False,
                               dtype={'patienthealthsystemstayid': str, "patientunitstayid": str})
        stays_df.columns = stays_df.columns.str.lower()

        # assert that there is no row with empty ICUSTAY_ID or HADM_ID
        assert(not stays_df['patientunitstayid'].isnull().any())
        assert(not stays_df['patienthealthsystemstayid'].isnull().any())
        
        # assert there are no repetitions of ICUSTAY_ID or HADM_ID
        # since admissions with multiple ICU stays were excluded
        assert(len(stays_df['patientunitstayid'].unique()) == len(stays_df['patientunitstayid']))
        assert(len(stays_df['patienthealthsystemstayid'].unique()) == len(stays_df['patienthealthsystemstayid']))
        
        if os.path.exists(os.path.join(args.subjects_root_path, subject, 'events.csv')) is False :
            shutil.rmtree(os.path.join(args.subjects_root_path, subject))
            print('removed subjects without events: ',subject) 
            continue
        events_df = pd.read_csv(os.path.join(args.subjects_root_path, subject, 'events.csv'), index_col=False,
                                dtype={'patienthealthsystemstayid': str, "patientunitstayid": str})
        events_df.columns = events_df.columns.str.lower()
      
        n_events += events_df.shape[0]

        empty_hadm += events_df['patienthealthsystemstayid'].isnull().sum()
        events_df = events_df.dropna(subset=['patienthealthsystemstayid'])
        merged_df = events_df.merge(stays_df, left_on=['patienthealthsystemstayid'], right_on=['patienthealthsystemstayid'],
                                    how='left', suffixes=['', '_r'], indicator=True)

      
        no_hadm_in_stay += (merged_df['_merge'] == 'left_only').sum()
        merged_df = merged_df[merged_df['_merge'] == 'both']
    
        cur_no_icustay = merged_df['patientunitstayid'].isnull().sum()
        no_icustay += cur_no_icustay
        merged_df.loc[:, 'patientunitstayid'] = merged_df['patientunitstayid'].fillna(merged_df['patientunitstayid_r'])
        recovered += cur_no_icustay - merged_df['patientunitstayid'].isnull().sum()
        could_not_recover += merged_df['patientunitstayid'].isnull().sum()
        merged_df = merged_df.dropna(subset=['patientunitstayid'])
        
    
        icustay_missing_in_stays += (merged_df['patientunitstayid'] != merged_df['patientunitstayid_r']).sum()
        merged_df = merged_df[(merged_df['patientunitstayid'] == merged_df['patientunitstayid_r'])]
        
        to_write = merged_df[['uniquepid', 'patienthealthsystemstayid', 'patientunitstayid', 'time', 'name', 'value', 'valueuom']]
        to_write.to_csv(os.path.join(args.subjects_root_path, subject, 'events.csv'), index=False)
           
    assert(could_not_recover == 0)
    print('n_events: {}'.format(n_events))
    print('empty_hadm: {}'.format(empty_hadm))
    print('no_hadm_in_stay: {}'.format(no_hadm_in_stay))
    print('no_icustay: {}'.format(no_icustay))
    print('recovered: {}'.format(recovered))
    print('could_not_recover: {}'.format(could_not_recover))
    print('icustay_missing_in_stays: {}'.format(icustay_missing_in_stays))


if __name__ == "__main__":
    main()
