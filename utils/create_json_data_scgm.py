"""
Creates json files in the Medical Segmentation Decathlon datalist format containing 
dictionaries of image-label pairs for training and testing for each center in the 
Spinal Cord Gray Matter (SCGM) dataset.
Website: http://niftyweb.cs.ucl.ac.uk/program.php?p=CHALLENGE
NOTE: On the challenge website, the centers are listed as 'UCL', 'EPM', 'VDB', 'UHZ'
we just use a slightly different naming convention here
"""

import os
import json
import argparse
from tqdm import tqdm
import numpy as np

root = "/Users/anonymous/projects/projects/ivadomed/gm_challenge_16_resampled"

parser = argparse.ArgumentParser(description='Code for creating data splits for each center')

parser.add_argument('-se', '--seed', default=42, type=int, help="Seed for reproducibility")
parser.add_argument('-dr', '--data_root', default=root, type=str, help='Path to the data set directory')
parser.add_argument('-ds', '--data_split', default='ucl', type=str, help='name of datasets to include', 
                    choices=['ucl', 'unf', 'vanderbilt', 'zurich', 'mix'])

args = parser.parse_args()

pth = "/Users/anonymous/continual-learning-medical/datalists/"
dataset_type = "scgm"
save_path = os.path.join(pth, dataset_type)

seed = args.seed
rng = np.random.default_rng(seed)
# random.seed(seed)

# Using 85-15 split for scgm 
fraction_test = 0.15

# create one json file with 80-20 train-test split
all_centers_subjects = os.listdir(args.data_root)

# only the first 10 subjects from ucl have the labels
ucl_subjects = [sub for sub in all_centers_subjects if sub.startswith('sub-ucl')][:10]
rng.shuffle(ucl_subjects)

# on the challenge website, this is under the name of "EPM"
unf_subjects = [sub for sub in all_centers_subjects if sub.startswith('sub-unf')]
rng.shuffle(unf_subjects)

# on the challenge website, this is under the name of "VDB"
vanderbilt_subjects = [sub for sub in all_centers_subjects if sub.startswith('sub-vanderbilt')]
rng.shuffle(vanderbilt_subjects)

# on the challenge website, this is under the name of "UHZ"
zurich_subjects = [sub for sub in all_centers_subjects if sub.startswith('sub-zurich')]
rng.shuffle(zurich_subjects)

if args.data_split == 'ucl':
    test_subjects = ucl_subjects[:int(len(ucl_subjects) * fraction_test)]
    # print('Held-out Subjects: ', test_subjects)

    # The rest of the subjects will be used for the train and validation phases
    training_subjects = ucl_subjects[int(len(ucl_subjects) * fraction_test):]

elif args.data_split == 'unf':
    test_subjects = unf_subjects[:int(len(unf_subjects) * fraction_test)]
    # print('Held-out Subjects: ', test_subjects)

    # The rest of the subjects will be used for the train and validation phases
    training_subjects = unf_subjects[int(len(unf_subjects) * fraction_test):]

elif args.data_split == 'vanderbilt':
    test_subjects = vanderbilt_subjects[:int(len(vanderbilt_subjects) * fraction_test)]
    # print('Held-out Subjects: ', test_subjects)

    # The rest of the subjects will be used for the train and validation phases
    training_subjects = vanderbilt_subjects[int(len(vanderbilt_subjects) * fraction_test):]

elif args.data_split == 'zurich':
    test_subjects = zurich_subjects[:int(len(zurich_subjects) * fraction_test)]
    # print('Held-out Subjects: ', test_subjects)

    # The rest of the subjects will be used for the train and validation phases
    training_subjects = zurich_subjects[int(len(zurich_subjects) * fraction_test):]

# Maybe dataset pairs could be used for pre-training. 
# TODO: Think about it!
elif args.data_split == 'mix':

    # ucl
    test_ucl = ucl_subjects[:int(len(ucl_subjects) * fraction_test)]
    # print('Held-out Subjects: ', test_ucl)
    training_ucl = ucl_subjects[int(len(ucl_subjects) * fraction_test):]

    # unf
    test_unf = unf_subjects[:int(len(unf_subjects) * fraction_test)]
    # print('Held-out Subjects: ', test_unf)
    training_unf = unf_subjects[int(len(unf_subjects) * fraction_test):]

    # vanderbilt
    test_vanderbilt = vanderbilt_subjects[:int(len(vanderbilt_subjects) * fraction_test)]
    training_vanderbilt = vanderbilt_subjects[int(len(vanderbilt_subjects) * fraction_test):]

    # zurich
    test_zurich = zurich_subjects[:int(len(zurich_subjects) * fraction_test)]
    training_zurich = zurich_subjects[int(len(zurich_subjects) * fraction_test):]
    
    test_subjects = test_ucl + test_unf + test_vanderbilt + test_zurich
    training_subjects = training_ucl + training_unf + training_vanderbilt + training_zurich


# keys to be defined in the dataset_0.json
params = {}
params["description"] = "CL for SCGM"
params["labels"] = {
    "0": "background",
    "1": "sc-gray-matter"
    }
params["seed_used"] = seed
params["modality"] = {
    "0": "MRI"
    }
params["name"] = f"continual-learning-scgm data"
params["numTest"] = len(test_subjects)
params["numTraining"] = len(training_subjects)
params["reference"] = "XX"
params["tensorImageSize"] = "3D"

train_subjects_dict = {"training": training_subjects,} 
test_subjects_dict =  {"test": test_subjects}

# run loop for training and validation subjects
for name, subs_list in train_subjects_dict.items():

    temp_list = []
    for subject_no, subject in enumerate(tqdm(subs_list, desc='Loading Volumes')):
        temp_data = {}        
        
        # Read-in input volumes
        t2star = os.path.join(args.data_root, subject, 'anat', f"{subject}_T2star.nii.gz")
        # every center has a GT label from unf so using that as the default
        gt = os.path.join(args.data_root, 'derivatives', 'labels', subject, 'anat', f"{subject}_T2star_gmseg-manual-unf.nii.gz")

        # store in a temp dictionary
        temp_data["image"] = t2star 
        temp_data["label"] = gt

        temp_list.append(temp_data)
    
    params[name] = temp_list


# run separte loop for testing
for name, subs_list in test_subjects_dict.items():
    temp_list = []
    for subject_no, subject in enumerate(tqdm(subs_list, desc='Loading Volumes')):
    
        temp_data = {}

        # Read-in input volumes format
        t2star = os.path.join(args.data_root, subject, 'anat', f"{subject}_T2star.nii.gz")
        
        # every center has a GT label from unf so using that as the default
        gt = os.path.join(args.data_root, 'derivatives', 'labels', subject, 'anat', f"{subject}_T2star_gmseg-manual-unf.nii.gz")

        # store in a temp dictionary
        temp_data["image"] = t2star 
        temp_data["label"] = gt
        
        temp_list.append(temp_data)
    
    params[name] = temp_list

final_json = json.dumps(params, indent=4, sort_keys=True)
jsonFile = open(os.path.join(save_path, f"dataset_{args.data_split}.json"), "w")
jsonFile.write(final_json)
jsonFile.close()





