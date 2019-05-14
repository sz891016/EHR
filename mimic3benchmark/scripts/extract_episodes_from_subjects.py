from __future__ import absolute_import
from __future__ import print_function

import argparse

import os
import sys

from mimic3benchmark.subject import read_stays, read_diagnoses, read_events, get_events_for_stay, add_hours_elpased_to_events
from mimic3benchmark.subject import convert_events_to_timeseries, get_first_valid_from_timeseries
from mimic3benchmark.preprocessing import read_itemid_to_variable_map, map_itemids_to_variables, read_variable_ranges, clean_events
from mimic3benchmark.preprocessing import transform_gender, transform_ethnicity, assemble_episodic_data


parser = argparse.ArgumentParser(description='Extract episodes from per-subject data.')
parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
parser.add_argument('--variable_map_file', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/itemid_to_variable_map.csv'),
                    help='CSV containing ITEMID-to-VARIABLE map.')
parser.add_argument('--reference_range_file', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/variable_ranges.csv'),
                    help='CSV containing reference ranges for VARIABLEs.')
parser.add_argument('--verbose', '-v', type=int, help='Level of verbosity in output.', default=1)
args, _ = parser.parse_known_args()

'''
得到ITEMID到临床变量VARIABLE的索引表
得到临床变量列表
'''
var_map = read_itemid_to_variable_map(args.variable_map_file)
variables = list(var_map.VARIABLE.unique())
#variables.remove('Capillary refill rate')
#variables.remove('Fraction inspired oxygen')
#variables.remove('pH')


for subject_dir in os.listdir(args.subjects_root_path):
    dn = os.path.join(args.subjects_root_path, subject_dir)
    try:
        subject_id = int(subject_dir)
        if not os.path.isdir(dn):
            raise Exception
    except:
        continue
    sys.stdout.write('Subject {}: '.format(subject_id))
    #flush()：刷新缓冲区，将缓冲区中的数据立刻写入文件，同时清空缓冲�?    
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
    
    #将该病人的stay和diagnoses合并到一起成为episodic_data
    episodic_data = assemble_episodic_data(stays, diagnoses)

    sys.stdout.write('cleaning and converting to time series...')
    sys.stdout.flush()
    #将events和var_map内容按照events里面的ITEMID为准合并�?    events = map_itemids_to_variables(events, var_map)
    #对events给定的列执行对应清洗函数
    events = map_itemids_to_variables(events, var_map)
    events = clean_events(events)
    if events.shape[0] == 0:
        sys.stdout.write('no valid events!\n')
        continue
    timeseries = convert_events_to_timeseries(events, variables=variables)
    
    sys.stdout.write('extracting separate episodes...')
    sys.stdout.flush()

    for i in range(stays.shape[0]):
        stay_id = stays.ICUSTAY_ID.iloc[i]
        sys.stdout.write(' {}'.format(stay_id))
        sys.stdout.flush()
        intime = stays.INTIME.iloc[i]
        outtime = stays.OUTTIME.iloc[i]

        episode = get_events_for_stay(timeseries, stay_id, intime, outtime)
        if episode.shape[0] == 0:
            sys.stdout.write(' (no data!)')
            sys.stdout.flush()
            continue
        ''' 添加新列：event[HOURS],将charttime列的数据删除,HOURS =(INTIME-CHARTTIME)/60*60
        '''
        episode = add_hours_elpased_to_events(episode, intime).set_index('HOURS').sort_index(axis=0)
        episodic_data.Weight.ix[stay_id] = get_first_valid_from_timeseries(episode, 'Weight')
        episodic_data.Height.ix[stay_id] = get_first_valid_from_timeseries(episode, 'Height')
        #有几次stay就有几个episode{}_timeseries.csv文件
        episodic_data.ix[episodic_data.index==stay_id].to_csv(os.path.join(args.subjects_root_path, subject_dir, 'episode{}.csv'.format(i+1)), index_label='Icustay')
        columns = list(episode.columns)
        columns_sorted = sorted(columns, key=(lambda x: "" if x == "Hours" else x))
        episode = episode[columns_sorted]
        episode.to_csv(os.path.join(args.subjects_root_path, subject_dir, 'episode{}_timeseries.csv'.format(i+1)), index_label='Hours')
    sys.stdout.write(' DONE!\n')
