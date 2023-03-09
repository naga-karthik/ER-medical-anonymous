import os
from monai.data import Dataset, load_decathlon_datalist
import nibabel as nib
from skimage.measure import shannon_entropy
import numpy as np

def get_test_datasets(dataset_type, dataset_path, test_centers, datalists_root, test_transforms, seed):
    """
    Get test datasets for each center in centers_list (used for testing across all centers).
    Depending on the dataset type, creates the corresponding MSD-style datalists on-the-fly
    and returns a list of Dataset objects.
    """
    datasets = []

    for center in test_centers:
        create_datalist_cmd = '%s %s -se %d -dr %s -ds %s'
        os.system(create_datalist_cmd % (
                            'python', f"./utils/create_json_data_{dataset_type}.py", seed, dataset_path, f"{center}"))    

        dataset_name = os.path.join(datalists_root, dataset_type, f"dataset_{center}.json")
        test_files = load_decathlon_datalist(dataset_name, True, "test")
        datasets.append(Dataset(data=test_files, transform=test_transforms))
    
    return datasets


def sort_by_entropy(prev_train_files, descending=False):
    """
    Sorts the training files by entropy (descending) and returns the sorted training files.
    """
    prev_train_files_sorted = []
    entropies_dict = {}
    
    for subject_files in prev_train_files:
        # get the image and label paths for this subject
        image_path = subject_files['image']
        label_path = subject_files['label']
        
        # load the label data
        label_data = nib.load(label_path).get_fdata()
        
        # calculate the entropy for the label data
        entropy = shannon_entropy(label_data.squeeze())
        
        # add the entropy to the dictionary
        entropies_dict[image_path] = (entropy, label_path)
        
    # sort the dictionary by entropy in descending/ascending order
    if descending:
        sorted_entropies = sorted(entropies_dict.items(), key=lambda item: item[1][0], reverse=True)
    else:
        sorted_entropies = sorted(entropies_dict.items(), key=lambda item: item[1][0])

    # create a new list of training files sorted by entropy
    for image_path, (entropy, label_path) in sorted_entropies:
        prev_train_files_sorted.append({'image': image_path, 'label': label_path})
        
    return prev_train_files_sorted
