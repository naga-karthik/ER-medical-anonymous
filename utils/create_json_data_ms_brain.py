"""
Creates json files in the Medical Segmentation Decathlon datalist format containing 
dictionaries of image-label pairs for training and testing for each center. 
"""

import os
import json
import argparse
import nibabel as nib
from tqdm import tqdm
import numpy as np

root = "/Users/anonymous/projects/ms_brain_spine/data_processing"

parser = argparse.ArgumentParser(description='Code for creating data splits for each center')

parser.add_argument('-se', '--seed', default=42, type=int, help="Seed for reproducibility")
parser.add_argument('-dr', '--data_root', default=root, type=str, help='Path to the data set directory')
parser.add_argument('-ds', '--data_split', default='BW', type=str, help='name of datasets to include', 
                    choices=['BW', 'RE', 'NI', 'AM', 'KA', 'MI', 'MO', 'UC', 'mix'])

args = parser.parse_args()

pth = "/Users/anonymous/continual-learning-medical/datalists/"
dataset_type = "ms_brain"
save_path = os.path.join(pth, dataset_type)

# print(save_path)

seed = args.seed
rng = np.random.default_rng(seed)
# random.seed(seed)

fraction_test = 0.2

# create one json file with 80-20 train-test split
all_centers_subjects = os.listdir(args.data_root)

# categoirizing all the subjects here
am_subjects = [sub for sub in all_centers_subjects if sub.startswith('AM')]
rng.shuffle(am_subjects)

bw_subjects = [sub for sub in all_centers_subjects if sub.startswith('BW')]
rng.shuffle(bw_subjects)

ka_subjects = [sub for sub in all_centers_subjects if sub.startswith('KA')]
rng.shuffle(ka_subjects)

# NOTE: MI has only T2 labels
mi_subjects = [sub for sub in all_centers_subjects if sub.startswith('MI')]
rng.shuffle(mi_subjects)

mo_subjects = [sub for sub in all_centers_subjects if sub.startswith('MO')]
rng.shuffle(mo_subjects)    

re_subjects = [sub for sub in all_centers_subjects if sub.startswith('RE')]
rng.shuffle(re_subjects)

ni_subjects = [sub for sub in all_centers_subjects if sub.startswith('NI')]
rng.shuffle(ni_subjects)

uc_subjects = [sub for sub in all_centers_subjects if sub.startswith('UC')]
rng.shuffle(uc_subjects)

if args.data_split == 'AM':
    test_subjects = am_subjects[:int(len(am_subjects) * fraction_test)]
    # print('Held-out Subjects: ', test_subjects)

    # The rest of the subjects will be used for the train and validation phases
    training_subjects = am_subjects[int(len(am_subjects) * fraction_test):]

elif args.data_split == 'BW':    
    test_subjects = bw_subjects[:int(len(bw_subjects) * fraction_test)]
    # print('Held-out Subjects: ', test_subjects)

    # The rest of the subjects will be used for the train and validation phases
    training_subjects = bw_subjects[int(len(bw_subjects) * fraction_test):]

elif args.data_split == 'KA':
    test_subjects = ka_subjects[:int(len(ka_subjects) * fraction_test)]
    # print('Held-out Subjects: ', test_subjects)

    # The rest of the subjects will be used for the train and validation phases
    training_subjects = ka_subjects[int(len(ka_subjects) * fraction_test):]

elif args.data_split == 'MI':
    test_subjects = mi_subjects[:int(len(mi_subjects) * fraction_test)]
    # print('Held-out Subjects: ', test_subjects)

    # The rest of the subjects will be used for the train and validation phases
    training_subjects = mi_subjects[int(len(mi_subjects) * fraction_test):]

elif args.data_split == 'MO':
    test_subjects = mo_subjects[:int(len(mo_subjects) * fraction_test)]
    # print('Held-out Subjects: ', test_subjects)

    # The rest of the subjects will be used for the train and validation phases
    training_subjects = mo_subjects[int(len(mo_subjects) * fraction_test):]

elif args.data_split == 'RE':
    # takes 20% of the rennes dataset as the held out test-set
    test_subjects = re_subjects[:int(len(re_subjects) * fraction_test)]
    # print('Held-out Subjects: ', test_subjects)

    # The rest of the subjects will be used for the train and validation phases
    training_subjects = re_subjects[int(len(re_subjects) * fraction_test):]

elif args.data_split == 'NI':
    # takes 20% of the rennes dataset as the held out test-set
    test_subjects = ni_subjects[:int(len(ni_subjects) * fraction_test)]
    # print('Held-out Subjects: ', test_subjects)

    # The rest of the subjects will be used for the train and validation phases
    training_subjects = ni_subjects[int(len(ni_subjects) * fraction_test):]

elif args.data_split == 'UC':
    # takes 20% of the rennes dataset as the held out test-set
    test_subjects = uc_subjects[:int(len(uc_subjects) * fraction_test)]
    # print('Held-out Subjects: ', test_subjects)

    # The rest of the subjects will be used for the train and validation phases
    training_subjects = uc_subjects[int(len(uc_subjects) * fraction_test):]

elif args.data_split == 'mix':

    # amu 
    test_amu = am_subjects[:int(len(am_subjects) * fraction_test)]
    training_amu = am_subjects[int(len(am_subjects) * fraction_test):]

    # bwh
    test_bwh = bw_subjects[:int(len(bw_subjects) * fraction_test)]
    training_bwh = bw_subjects[int(len(bw_subjects) * fraction_test):]

    # karo
    test_karo = ka_subjects[:int(len(ka_subjects) * fraction_test)]
    training_karo = ka_subjects[int(len(ka_subjects) * fraction_test):]

    # milan
    test_milan = mi_subjects[:int(len(mi_subjects) * fraction_test)]
    training_milan = mi_subjects[int(len(mi_subjects) * fraction_test):]

    # montpellier
    test_montpellier = mo_subjects[:int(len(mo_subjects) * fraction_test)]
    training_montpellier = mo_subjects[int(len(mo_subjects) * fraction_test):]

    # rennes
    test_rennes = re_subjects[:int(len(re_subjects) * fraction_test)]
    training_rennes = re_subjects[int(len(re_subjects) * fraction_test):]

    # nih
    test_nih = ni_subjects[:int(len(ni_subjects) * fraction_test)]
    training_nih = ni_subjects[int(len(ni_subjects) * fraction_test):]

    # ucsf
    test_ucsf = uc_subjects[:int(len(uc_subjects) * fraction_test)]
    training_ucsf = uc_subjects[int(len(uc_subjects) * fraction_test):]


    test_subjects = test_amu + test_bwh + test_karo + test_milan + test_montpellier + test_rennes + test_nih + test_ucsf
    training_subjects = training_amu + training_bwh + training_karo + training_milan + \
                             training_montpellier + training_rennes + training_nih + training_ucsf


# keys to be defined in the dataset_0.json
params = {}
params["description"] = "CL for MS"
params["labels"] = {
    "0": "background",
    "1": "ms-lesion"
    }
params["seed_used"] = seed
params["modality"] = {
    "0": "MRI"
    }
params["name"] = f"continual-learning-ms data"
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
        
        if subject.startswith('MI'):
            # Turns out it has some corrupted data. check by loading with nibabel before creating datalist
            # Read-in input volumes
            t2 = os.path.join(args.data_root, subject, 'brain', 't2', 't2_mni.nii.gz')
            # Read-in GT volumes
            gtc = os.path.join(args.data_root, subject, 'brain', 't2', 't2_lesion_manual_mni.nii.gz')

            try:
                nib.load(t2).get_fdata()
                # store in a temp dictionary
                temp_data["image"] = t2 
                temp_data["label"] = gtc 
            except Exception as e:
                print(f"Subject {subject}'s data is corrupted, skipping subject!")
                continue

        
        elif subject.startswith('KA'):
            image = os.path.join(args.data_root, subject, 'brain', 't2', 't2_mni.nii.gz')
            if os.path.exists(image):
                # if T2 image exists (only for some subjects), then use that image-label pair
                temp_data["image"] = image
                temp_data["label"] = os.path.join(args.data_root, subject, 'brain', 't2', 't2_lesion_manual_mni.nii.gz')
            else:
                # if not, then use the flair image-label pair (true for some subjects)
                temp_data["image"] = os.path.join(args.data_root, subject, 'brain', 'flair', 'flair_mni.nii.gz')
                temp_data["label"] = os.path.join(args.data_root, subject, 'brain', 'flair', 'flair_lesion_manual_mni.nii.gz')

        else:
            # Read-in input volumes
            ses01_flair = os.path.join(args.data_root, subject, 'brain', 'flair', 'flair_mni.nii.gz')
            # Read-in GT volumes
            gtc = os.path.join(args.data_root, subject, 'brain', 'flair', 'flair_lesion_manual_mni.nii.gz')
            
            # store in a temp dictionary
            temp_data["image"] = ses01_flair 
            temp_data["label"] = gtc 

        temp_list.append(temp_data)
    
    params[name] = temp_list


# run separte loop for testing
for name, subs_list in test_subjects_dict.items():
    temp_list = []
    for subject_no, subject in enumerate(tqdm(subs_list, desc='Loading Volumes')):
    
        temp_data = {}

        if subject.startswith('MI'):
            # Read-in input volumes
            t2 = os.path.join(args.data_root, subject, 'brain', 't2', 't2_mni.nii.gz')
            # Read-in GT volumes
            gtc = os.path.join(args.data_root, subject, 'brain', 't2', 't2_lesion_manual_mni.nii.gz')
            
            # store in a temp dictionary
            temp_data["image"] = t2 
            temp_data["label"] = gtc 
        
        elif subject.startswith('KA'):
            image = os.path.join(args.data_root, subject, 'brain', 't2', 't2_mni.nii.gz')
            if os.path.exists(image):
                # if T2 image exists (only for some subjects), then use that image-label pair
                temp_data["image"] = image
                temp_data["label"] = os.path.join(args.data_root, subject, 'brain', 't2', 't2_lesion_manual_mni.nii.gz')
            else:
                # if not, then use the flair image-label pair (true for some subjects)
                temp_data["image"] = os.path.join(args.data_root, subject, 'brain', 'flair', 'flair_mni.nii.gz')
                temp_data["label"] = os.path.join(args.data_root, subject, 'brain', 'flair', 'flair_lesion_manual_mni.nii.gz')

        else:
            # Read-in input volumes
            ses01_flair = os.path.join(args.data_root, subject, 'brain', 'flair', 'flair_mni.nii.gz')
            # Read-in GT volumes
            gtc = os.path.join(args.data_root, subject, 'brain', 'flair', 'flair_lesion_manual_mni.nii.gz')

            # store in a temp dictionary
            temp_data["image"] = ses01_flair 
            temp_data["label"] = gtc

        temp_list.append(temp_data)
    
    params[name] = temp_list

final_json = json.dumps(params, indent=4, sort_keys=True)
jsonFile = open(os.path.join(save_path, f"dataset_{args.data_split}.json"), "w")
jsonFile.write(final_json)
jsonFile.close()





