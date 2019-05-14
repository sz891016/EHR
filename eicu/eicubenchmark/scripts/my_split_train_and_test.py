from __future__ import absolute_import
from __future__ import print_function

import os
import shutil
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split



def move_to_partition(args, patients, partition):
    if not os.path.exists(os.path.join(args.subjects_root_path, partition)):
        os.mkdir(os.path.join(args.subjects_root_path, partition))
    for patient in patients:
        src = os.path.join(args.subjects_root_path, patient)
        dest = os.path.join(args.subjects_root_path, partition, patient)
        shutil.move(src, dest)


def main():
    parser = argparse.ArgumentParser(description='Split data into train and test sets.')
    parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
    args, _ = parser.parse_known_args()
    
    data = pd.read_csv('data/in-hospital-mortality/listfile.csv')
    x,y = data.ix[:,0],data.ix[:,1]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
 
    train_patients=list(x_train)
    test_patients=list(x_test)
    assert len(set(train_patients) & set(test_patients)) == 0

    move_to_partition(args, train_patients, "train")
    move_to_partition(args, test_patients, "test")

if __name__ == '__main__':
    main()