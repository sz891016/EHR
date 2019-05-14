from __future__ import absolute_import
from __future__ import print_function

import csv
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Extract per-subject data from EICU CSV files.')
parser.add_argument('subject_path', type=str, help='Directory containing all subjects files.')
args, _ = parser.parse_known_args()

stays = pd.read_csv(os.path.join(args.subject_path,'all_stays.csv'))
stays = stays[['uniquepid','MORTALITY']].drop_duplicates(subset=['uniquepid'],keep='last')
stays.to_csv(os.path.join('eicubenchmark/resources/','eicutestset.csv'),index = False)

 


