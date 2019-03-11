# MESC-Net

This is a python demo for the paper:<br />
**Chuyang Ye et al., "A Deep Network for Tissue Microstructure Estimation Using Modified LSTM Units", under review in MedIA.** 

The demo includes both the training and test phase. Therefore, to run it, both the training and test data (which are images in the NIfTI format) should be prepared. The input diffusion signals should be normalized.

There are a few dependencies that need to be installed:<br />
**numpy <br />
nibabel <br />
keras <br />
theano <br />**

Here is how to run the scripts <br />
>python MESCNet.py < list of training normalized diffusion images> < list of training brain mask images > < number of microstructure meaasures to be estimated > < list of training microstructure 1 > ... < list of training microstructure N > < list of test normalized diffusion images > < list of test brain mask images > < output directory > <br />

For example, <br />
>python MESCNet.py dwis_training.txt masks_training.txt 3 icvfs_training.txt isos_training.txt ods_training.txt dwis_test.txt masks_test.txt output

For more questions, please contact me via chuyang.ye@bit.edu.cn or pkuclosed@gmail.com
