# Azubi-Hackathon-Project
Below you can find a outline of how to reproduce my solution for the azubi africa competition.
If you run into any trouble with the setup/code or have any questions please contact me us at b.korojo@gmail.com

#ARCHIVE CONTENTS
model.rar          : original model upload - contains original code, additional training examples, corrected labels, etc
comp_etc                     : contains ancillary information for prediction - clustering of training/test examples
comp_mdl                     : model binaries used in generating solution
comp_preds                   : model predictions
train_code                  : code to rebuild models from scratch
predict_code                : code to generate predictions from model binaries

#HARDWARE: (The following specs were used to create the original solution)
Edition Windows 10 Pro - Version 2004 - OS build 19041.450 (256 GB boot disk)
8 core CPU, 16 GB memory
1 x NVIDIA 940mx

#SOFTWARE (python packages are detailed separately in `requirements.txt`):
Python 3.7.8
CUDA 11.0.228
nvidia drivers v.3.20.4.14

#DATA SETUP 
./Azubi Africa/data/raw/stage1/label_data.csv


#DATA PROCESSING
#The train/predict code will also call this script if it has not already been run on the relevant data.
python ./train_code/prepare_data.py --data_dir=data/stage1/ --output_dir=data/stage1_cleaned

#MODEL BUILD
This model is build with the MaxAbsScaler and AdaBoost Classifier
The prediction runs in a few minutes and uses precomputed model predictions.
Saved model binaries stored in Azubi Afirca/submissions/comp_mdl
