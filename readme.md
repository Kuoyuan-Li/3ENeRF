# 3DMM-guided Efficient Neural Radiance Field 

This repository contains the implementation of 3ENeRF based on Instant-NSR-PL, 3DMM and the Pytorch-Lightning framework.

This repository carries out the work I have done during my year-long research project as a part of my Master of Computer Science degree at the University of Melbourne. The project is supervised by Dr. Mingming Gong and Dr. Kris Ehinger.


## To run the code

### 1. Prepare the dataset
Please refer to the data_prepare_readme.md in the datasets folder to prepare the dataset. Note: the guidance is expected to be run within the University of Melbourne's Spartan HPC.

### 2. Train the model
The main entry point for training the model is `launch.py` in the instant-nsr-pl folder. The script is designed to be run on the Spartan HPC with Slurm script.

