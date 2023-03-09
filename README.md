# ER-medical-anonymous

### Structure of the Repository

1. `main_pl_*.py`: These files contain the main code for the four types of experiments, each having a separate file. 

2. `scripts/`: Contains the bash scripts for calling each of the `main_pl_*.py` files to train the model depending on the type of experiment across multiple seeds. Also, contains the script preprocessing the public dataset used in the paper.

3. `utils/`: Contains 4 files

    a. `create_json_data_*.py`: Creates a `json` file (in the Medical Segmentation Decathlon format) for both datasets used in the paper

    b. `data_utils.py`: Contains the necessary functions for loading the data, creating the dataloaders, and entropy-based sample selection. 

    c. `metrics.py`: Contains the implementations of continual learning metrics. 

    d. `transforms.py`: Contains the necessary functions for data augmentation for both datasets.