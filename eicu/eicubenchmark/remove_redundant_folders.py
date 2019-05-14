#remove folders from root which is not in testset
import os
import shutil
import argparse

parser = argparse.ArgumentParser(description='Split data into train and test sets.')
parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
args, _ = parser.parse_known_args()

nameset = set()
with open(os.path.join(os.path.dirname(__file__), './resources/testset.csv'), "r") as test_set_file:
        for line in test_set_file:
            x,y = line.split(',')
            x = nameset.add(x)
print('====>nameset',len(nameset))
            
folders = os.listdir(args.subjects_root_path)
folders = list(filter(str.isdigit,folders))
print('====>folders',len(folders))

i=0
for f in folders:
    if f not in nameset:
        shutil.rmtree(os.path.join(args.subjects_root_path,f))
        i=i+1
        
print(i)

