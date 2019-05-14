from __future__ import absolute_import
from __future__ import print_function

import argparse
import yaml
import os

from mimic3benchmark.mimic3csv import *
from mimic3benchmark.preprocessing import add_hcup_ccs_2015_groups, make_phenotype_label_matrix
from mimic3benchmark.util import *

'''
hcup_ccs_2015_definitions.yaml:所有疾病的ICD-9诊断代码
benchmark中只选取了常见的25种疾病
'''
parser = argparse.ArgumentParser(description='Extract per-subject data from MIMIC-III CSV files.')
parser.add_argument('mimic3_path', type=str, help='Directory containing MIMIC-III CSV files.')
parser.add_argument('output_path', type=str, help='Directory where per-subject data should be written.')
parser.add_argument('--event_tables', '-e', type=str, nargs='+', help='Tables from which to read events.',
                    default=['CHARTEVENTS', 'LABEVENTS', 'OUTPUTEVENTS'])
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

#得到病人基本信息，入院信息，ICU入住信息
patients = read_patients_table(args.mimic3_path)
admits = read_admissions_table(args.mimic3_path)
stays = read_icustays_table(args.mimic3_path)
if args.verbose:
    #shape[1] 为第一维（行）的长度，.shape[0] 为第二维（列）的长度
    print('START:', stays.ICUSTAY_ID.unique().shape[0], stays.HADM_ID.unique().shape[0],
          stays.SUBJECT_ID.unique().shape[0])

#去掉有ICU转院记录的stay
stays = remove_icustays_with_transfers(stays)
if args.verbose:
    print('REMOVE ICU TRANSFERS:', stays.ICUSTAY_ID.unique().shape[0], stays.HADM_ID.unique().shape[0],
          stays.SUBJECT_ID.unique().shape[0])

#合并一次住院有多次ICU入住的stay
#合并一个病人有多次ICU入住的stay
stays = merge_on_subject_admission(stays, admits)
stays = merge_on_subject(stays, patients)
stays = filter_admissions_on_nb_icustays(stays)
if args.verbose:
    print('REMOVE MULTIPLE STAYS PER ADMIT:', stays.ICUSTAY_ID.unique().shape[0], stays.HADM_ID.unique().shape[0],
          stays.SUBJECT_ID.unique().shape[0])

#将年龄AGE计算出来，加入到stay
#计算出入住ICU时病人生存状态：死亡(1) 其他(0)
#计算出入院时病人生存状态：死亡(1) 其他(0)
#去除不符合年龄要求的stay(>18)
stays = add_age_to_icustays(stays)
stays = add_inunit_mortality_to_icustays(stays)
stays = add_inhospital_mortality_to_icustays(stays)
stays = filter_icustays_on_age(stays)
if args.verbose:
    print('REMOVE PATIENTS AGE < 18:', stays.ICUSTAY_ID.unique().shape[0], stays.HADM_ID.unique().shape[0],
          stays.SUBJECT_ID.unique().shape[0])
'''
to_csv是DataFrame类下的一个方法
stays是DataFrame一个实例
第一个参数是输出路径
index：是否保留行索引，default:true 
'''
stays.to_csv(os.path.join(args.output_path, 'all_stays.csv'), index=False)
diagnoses = read_icd_diagnoses_table(args.mimic3_path)
#去除stay和diagnoses中病人ID和住院ID相同的行
diagnoses = filter_diagnoses_on_stays(diagnoses, stays)
#生成所有的诊断信息
diagnoses.to_csv(os.path.join(args.output_path, 'all_diagnoses.csv'), index=False)
#统计每个ICD-9代码出现的次数
count_icd_codes(diagnoses, output_path=os.path.join(args.output_path, 'diagnosis_counts.csv'))

'''
制作表型标签矩阵
phenotypes:比diagnoses多了：'HCUP_CCS_2015'与'USE_IN_BENCHMARK'列
'''
phenotypes = add_hcup_ccs_2015_groups(diagnoses, yaml.load(open(args.phenotype_definitions, 'r')))
make_phenotype_label_matrix(phenotypes, stays).to_csv(os.path.join(args.output_path, 'phenotype_labels.csv'),
                                                      index=False, quoting=csv.QUOTE_NONNUMERIC)
#test数据集随机选1000个病人
'''
iloc——通过行号索引行数据
merge():合并'SUBJECT_ID'相同的行
'''
if args.test:
    pat_idx = np.random.choice(patients.shape[0], size=1000)
    patients = patients.iloc[pat_idx]
    stays = stays.merge(patients[['SUBJECT_ID']], left_on='SUBJECT_ID', right_on='SUBJECT_ID')
    args.event_tables = [args.event_tables[0]]
    print('Using only', stays.shape[0], 'stays and only', args.event_tables[0], 'table')

'''
取出SUBJECT_ID不同的stay
将stays按照SUBJECT_ID分开，生成stays.csv，存放在以病人id命名的文件夹下，包含该病人所有的住院基本信息
将diagnoses按照SUBJECT_ID分开，生成diagnoses.csv，存放在以病人id命名的文件夹下，包含该病人所有的诊断信息
'''
subjects = stays.SUBJECT_ID.unique()
break_up_stays_by_subject(stays, args.output_path, subjects=subjects, verbose=args.verbose)
break_up_diagnoses_by_subject(phenotypes, args.output_path, subjects=subjects, verbose=args.verbose)

#通过包含itemid的表来选出用到的itemid
items_to_keep = set(
    [int(itemid) for itemid in dataframe_from_csv(args.itemids_file)['ITEMID'].unique()]) if args.itemids_file else None
'''
对event时间表读取内容，且按照SUBJECT_ID分开
每个病人有多少行事件，最多处理100000行
'''
for table in args.event_tables:
    read_events_table_and_break_up_by_subject(args.mimic3_path, table, args.output_path, items_to_keep=items_to_keep,
                                              subjects_to_keep=subjects, verbose=args.verbose)
