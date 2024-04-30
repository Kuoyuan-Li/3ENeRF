# How to prepare data for instant-portrait

## 1. Film the video with the phone, in 2 stages:
- Stage 1: the person is static and moving the camera
- Stage 2: the person is moving head (with no large movement) and the camera is moving

## 2. Extract frames from the video
- Using capture2images.ipynb

## 3. Running DECA deform_construct.slurm to get the DECA meshes


## 4. Running COLMAP to get the camera pose

Using the DECA meshes to create mask and mask out moving part

Then, using COLMAP to get the camera pose, following:
1. module load gcccore/10.2.0 apptainer/1.0.3
2. module load cuda/11.1.1
3. apptainer exec --nv -B /data:/data /data/scratch/common/singularity_images/colmap/colmap-3.9.sif colmap automatic_reconstructor --workspace_path ./colmap --image_path ./rgb/ds --mask_path ./rgb/face_masks --data_type video --quality medium --single_camera 1

Note: please `cd` to the correct directory (under dataset name) before running the command

## 5. Running processing(old).ipynb to do triangulation and get the mesh

## 6. Running deca2colmap.ipynb to get the meshes for all frames


# Assertion

## Using mesh_alignment_assertion.ipynb to check the alignment of the meshes

