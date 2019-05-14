from __future__ import absolute_import
from __future__ import print_function

import argparse

import os
import sys

from eicubenchmark.subject import read_stays, read_diagnoses, read_events, read_variables,get_events_for_stay, add_hours_elpased_to_events
from eicubenchmark.subject import convert_events_to_timeseries, get_first_valid_from_timeseries
from eicubenchmark.preprocessing import transform_gender, transform_ethnicity, assemble_episodic_data,make_variable_csv,clean_events

parser = argparse.ArgumentParser(description='Extract episodes from per-subject data.')
parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
parser.add_argument('--variable_map_file', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/'),
                    help='CSV containing variables.')                   
parser.add_argument('--reference_range_file', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/variable_ranges.csv'),
                    help='CSV containing reference ranges for VARIABLEs.')
parser.add_argument('--verbose', '-v', type=int, help='Level of verbosity in output.', default=1)
args, _ = parser.parse_known_args()


for subject_dir in os.listdir(args.subjects_root_path):
    dn = os.path.join(args.subjects_root_path, subject_dir)
    try:
        subject_id = int(subject_dir)
        if not os.path.isdir(dn):
            raise Exception
    except:
        continue
    sys.stdout.write('Subject {}: '.format(subject_id))
    sys.stdout.flush()

    try:
        sys.stdout.write('reading...')
        sys.stdout.flush()
        
        stays = read_stays(os.path.join(args.subjects_root_path, subject_dir))
        diagnoses = read_diagnoses(os.path.join(args.subjects_root_path, subject_dir))
        events = read_events(os.path.join(args.subjects_root_path, subject_dir))
    except:
        sys.stdout.write('error reading from disk!\n')
        continue
    else:
        sys.stdout.write('got {0} stays, {1} diagnoses, {2} events...'.format(stays.shape[0], diagnoses.shape[0], events.shape[0]))
        sys.stdout.flush()

    episodic_data = assemble_episodic_data(stays, diagnoses)
    
    sys.stdout.write('cleaning and converting to time series...')
    sys.stdout.flush()
    
    events = clean_events(events)
    if events.shape[0] == 0:
        sys.stdout.write('no valid events!\n')
        continue
    variables = ['Diastolic blood pressure','Glascow coma scale eye opening','Glascow coma scale motor response','Glascow coma scale total','Glascow coma scale verbal response','Glucose','Heart Rate','Height','Mean blood pressure','Oxygen saturation','Respiratory rate','Systolic blood pressure','Temperature','Weight']
    timeseries = convert_events_to_timeseries(events, variables = variables)
    
    sys.stdout.write('extracting separate episodes...')
    sys.stdout.flush()
    
    for i in range(stays.shape[0]):
        stay_id = stays.patientunitstayid.iloc[i]
        sys.stdout.write(' {}'.format(stay_id))
        sys.stdout.flush()
        intime = stays.INTIME.iloc[i]
        outtime = stays.OUTTIME.iloc[i]
        
        episode = get_events_for_stay(timeseries, stay_id,  outtime)
        
        if episode.shape[0] == 0:
            sys.stdout.write(' (no data!)')
            sys.stdout.flush()
            continue
        
        episode = add_hours_elpased_to_events(episode, intime).set_index('HOURS').sort_index(axis=0)
        episodic_data.ix[episodic_data.index==stay_id].to_csv(os.path.join(args.subjects_root_path, subject_dir, 'episode{}.csv'.format(i+1)), index_label='Icustay')
        columns = list(episode.columns)
        columns_sorted = sorted(columns, key=(lambda x: "" if x == "Hours" else x))
        episode = episode[columns_sorted]
        episode.to_csv(os.path.join(args.subjects_root_path, subject_dir, 'episode{}_timeseries.csv'.format(i+1)), index_label='Hours')
    sys.stdout.write(' DONE!\n')