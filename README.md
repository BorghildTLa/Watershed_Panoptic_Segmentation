# Watershed Panoptic Segmentation
This repository contains codes for training and testing a watershed-based panoptic segmentation deep-learning model. 
- **training:** For training the model
- **prediction:** For testing a trained model on new slides
# Data
Whole slide DN biopsies can be found at [athena.rc.ufl.edu](https://athena.rc.ufl.edu/)

Trained model files can be found in *[this box folder](https://trailblazer.box.com/s/r4zsgbffmfbrw7gp0flhh82ez0vq44xe)*

# Requirements
- Tensorflow 1.14.0
- Python 3.6.5
- OpenSlide 3.4.0
- joblib 0.11
- imgaug 0.4.0
- imageio 2.3.0
- opencv 3.4.0
- PIL 5.3.0

# Usage
## New Project Creation
From either the training or prediction folder, run the following code to create a new project:
```
python3 segmentation_school.py --option new --project name_of_project/ --base_dir /where/to/save/project/folders/
```
Once the project folders have been generated, place whole slide images in:

*/base_dir/project/TRAINING_data/0/*
## Network Training
Edit the "segmentation_school.py" script to configure training parameters, or include in command line when running.

Place whole slide images for training in:

*/base_dir/project/TRAINING_data/0/*

ResNet50 pretrained checkpoint file can be found at *box*

Place ResNet50 file in:

*/base_dir/project/MODELS/0/HR/*

From the training folder, run the script:
```
python3 segmentation_school.py --option train --project name_of_project/ --base_dir /where/to/save/project/folders/ --wsi_ext '.svs,.ndpi,...' --classNum X --one_network True 
```
Include any other arguments you wish to change from default at the end of this line (training steps, learning rates, etc.), or change directly in the python script
## Network Prediction
Edit the "segmentation_school.py" script to configure testing parameters, or include in command line when running.

Place whole slide images for testing in:

*/base_dir/project/TRAINING_data/1/* or other highest number available

Place model files for testing in:

*/base_dir/project/MODELS/1/HR/* or other highest number matching TRAINING_data folder

Pretrained model files for segmenting 6 renal compartments can be found at *box*

From the prediction folder, run the script:
```
python3 segmentation_school.py --option predict --project project_name/ --base_dir /where/to/save/project/folders/ --wsi_ext '.svs,.ndpi,...' --classNum X --one_network True
```
For segmentation on 6 renal compartments using pretrained model, use:
```
--classNum 6
```
Include any other arguments you wish to change from default at the end of this line (box size, etc.), or change directly in the python script
# Acknowledgements
Please cite our work!
