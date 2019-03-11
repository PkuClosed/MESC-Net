import sys
import os
import nibabel as nib
import numpy as np
from keras.models import Sequential, Model
from keras.layers.core import Lambda, Dropout
from keras.optimizers import SGD, Adam
from keras.layers.advanced_activations import ThresholdedReLU
from keras.layers import merge, Dense, Input, Activation, add, multiply
from keras.constraints import nonneg
from keras.layers.merge import add
import time

#%%        
dwinames = sys.argv[1]
masknames = sys.argv[2]
featurenumbers = int(sys.argv[3])
featurenames = []
for feature_index in range(featurenumbers):
    featurenames.append(sys.argv[4 + feature_index])
testdwinames = sys.argv[4 + featurenumbers]
testmasknames = sys.argv[5 + featurenumbers]
directory = sys.argv[6 + featurenumbers]

if os.path.exists(directory) == False:
    os.mkdir(directory)

#%%
start = time.time()
###### Training #######
print "Training Phase"    

#### load images
print "Loading"    

with open(dwinames) as f:
    allDwiNames = f.readlines()
with open(masknames) as f:
    allMaskNames = f.readlines()
allFeatureNames = []
for feature_index in range(featurenumbers):
    tempFeatureNames = None
    with open(featurenames[feature_index]) as f:
        tempFeatureNames = f.readlines()
    allFeatureNames.append(tempFeatureNames)
allDwiNames = [x.strip('\n') for x in allDwiNames]
allMaskNames = [x.strip('\n') for x in allMaskNames]
for feature_index in range(featurenumbers):
    allFeatureNames[feature_index] = [x.strip('\n') for x in allFeatureNames[feature_index]]

#%%
### setting voxels ###
Np = 10
nVox = 0
for iMask in range(len(allMaskNames)):
    print "Counting Voxels for Subject", iMask
    mask = nib.load(allMaskNames[iMask]).get_data()
    # number of voxels
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                if mask[i,j,k] > 0:
                    nVox = nVox + 1
     
dwi = nib.load(allDwiNames[0]).get_data()                   
featureTraining = np.zeros([nVox, featurenumbers])
neighbor_size = 27        
dwiTraining = np.zeros([nVox, dwi.shape[3]*neighbor_size])



print "Initializing Voxel List"

nVox = 0
    
for iMask in range(len(allDwiNames)):
    print "Setting Voxel List for Subject:", iMask
    dwi_nii = nib.load(allDwiNames[iMask])
    dwi = dwi_nii.get_data()
    mask = nib.load(allMaskNames[iMask]).get_data()
    feature = []
    for feature_index in range(featurenumbers):
        tempFeature = nib.load(allFeatureNames[feature_index][iMask]).get_data()
        feature.append(tempFeature)
    # number of voxels
    for i in range(dwi.shape[0]):
        for j in range(dwi.shape[1]):
            for k in range(dwi.shape[2]):
                if mask[i,j,k] > 0:
                    for ii in [-1,0,1]:
                        for jj in [-1,0,1]:
                            for kk in [-1,0,1]:
                                if i + ii >= 0 and i + ii < dwi.shape[0] and j + jj >= 0 and j + jj < dwi.shape[1] \
                                    and k + kk >= 0 and k + kk < dwi.shape[2] and mask[i + ii, j + jj, k + kk] > 0:  
                                    dwiTraining[nVox, ((ii+1)*9 + (jj+1)*3 + (kk+1))*dwi.shape[3]:((ii+1)*9 + (jj+1)*3 + (kk+1) + 1)*dwi.shape[3]] = dwi[i+ii, j+jj, k+kk, :]
                                else:
                                    dwiTraining[nVox, ((ii+1)*9 + (jj+1)*3 + (kk+1))*dwi.shape[3]:((ii+1)*9 + (jj+1)*3 + (kk+1) + 1)*dwi.shape[3]] = dwi[i, j, k, :]
                    for feature_index in range(featurenumbers):
                        featureTraining[nVox,feature_index] = feature[feature_index][i,j,k]
                    nVox = nVox + 1
#%%
means = np.mean(featureTraining, axis = 0)               
scales = np.log10(means)
scalesint = np.floor(scales)
featureTraining = featureTraining/np.power(10,scalesint)
print "scales:", scalesint
#%%
### setting architechture ###                    
print "Setting Architechture"
nDict1 = 301
nDict2 = 75

nLayers1 = 8
nLayers2 = 3
tau = 1e-10

ReLUThres = 0.01
tau = 1e-10

y = Input(shape=(dwiTraining.shape[1],))

Wy = Dense(nDict1, activation='linear', use_bias = True)(y)
Wfy = Dense(nDict1, activation='linear', use_bias = True)(y)
Wiy = Dense(nDict1, activation='linear', use_bias = True)(y)

W_fx = Sequential()
W_fx.add(Dense(nDict1, activation='linear', use_bias = True, input_shape=(nDict1,)))
W_ix = Sequential()
W_ix.add(Dense(nDict1, activation='linear', use_bias = True, input_shape=(nDict1,)))
S = Sequential()
S.add(Dense(nDict1, activation='linear', use_bias = True, input_shape=(nDict1,)))

## initialize \tilde{C}, x, and C for t = 1

Ctilde = Wy 

I = Activation('sigmoid')(Wiy) # x^{0} = 0

C = multiply([I, Ctilde]) # c^{0} = 0

x = ThresholdedReLU(theta = ReLUThres)(C)

nLayers = 8
for l in range(nLayers-1):
    Ctilde = add([Wy, S(x)])
    
    Wfx_Wfy = add([W_fx(x), Wfy])
    F = Activation('sigmoid')(Wfx_Wfy) 
    Wix_Wiy = add([W_ix(x), Wiy])
    I = Activation('sigmoid')(Wix_Wiy) 
    Cf = multiply([F, C])
    Ci = multiply([I, Ctilde])
    C = add([Cf, Ci])
    
    x = ThresholdedReLU(theta = ReLUThres)(C)

H = Sequential()
H.add(Dense(nDict2, input_dim = nDict1, activation='relu'))
for i in range(nLayers2-1):
    H.add(Dense(nDict2, activation='relu'))
H.add(Dense(featurenumbers, activation='relu'))

outputs = H(x)


epoch = 10

print "nLayers1, nLayers2, ReLUThres, epoch, nDict1, nDict2: ", \
nLayers1, nLayers2, ReLUThres, epoch, nDict1, nDict2

### fitting the model ###                    
print "Fitting"    

clf = Model(inputs=y, outputs=outputs)
clf.compile(optimizer=Adam(lr=0.0001), loss='mse')
print clf.summary()

hist = clf.fit(dwiTraining, featureTraining, batch_size=128, epochs=epoch, verbose=1, validation_split=0.1)
print(hist.history)
end = time.time()
print "Training took ", (end-start)

#%%###### Test #######
print "Test Phase"    

start = time.time()
with open(testdwinames) as f:
    allTestDwiNames = f.readlines()
with open(testmasknames) as f:
    allTestMaskNames = f.readlines()

allTestDwiNames = [x.strip('\n') for x in allTestDwiNames]
allTestMaskNames = [x.strip('\n') for x in allTestMaskNames]


for iMask in range(len(allTestDwiNames)):
    print "Processing Subject: ", iMask
    #### load images
    print "Loading"  
    dwi_nii = nib.load(allTestDwiNames[iMask])
    dwi = dwi_nii.get_data()
    mask = nib.load(allTestMaskNames[iMask]).get_data()
    print "Counting Voxels"
    nVox = 0
    for i in range(dwi.shape[0]):
        for j in range(dwi.shape[1]):
            for k in range(dwi.shape[2]):
                if mask[i,j,k] > 0:
                    nVox = nVox + 1
                    
    voxelList = np.zeros([nVox, 3], int)
    dwiTest = np.zeros([nVox, dwi.shape[3]*neighbor_size])

    print "Setting Voxels"
    nVox = 0
    for i in range(dwi.shape[0]):
        for j in range(dwi.shape[1]):
            for k in range(dwi.shape[2]):
                if mask[i,j,k] > 0:
                    voxelList[nVox,0] = i
                    voxelList[nVox,1] = j
                    voxelList[nVox,2] = k
                    for ii in [-1,0,1]:
                        for jj in [-1,0,1]:
                            for kk in [-1,0,1]:
                                if i + ii >= 0 and i + ii < dwi.shape[0] and j + jj >= 0 and j + jj < dwi.shape[1] \
                                    and k + kk >= 0 and k + kk < dwi.shape[2] and mask[i + ii, j + jj, k + kk] > 0:  
                                    dwiTest[nVox, ((ii+1)*9 + (jj+1)*3 + (kk+1))*dwi.shape[3]:((ii+1)*9 + (jj+1)*3 + (kk+1) + 1)*dwi.shape[3]] = dwi[i+ii,j+jj,k+kk,:]
                                else:
                                    dwiTest[nVox, ((ii+1)*9 + (jj+1)*3 + (kk+1))*dwi.shape[3]:((ii+1)*9 + (jj+1)*3 + (kk+1) + 1)*dwi.shape[3]] = dwi[i,j,k,:]
                    nVox = nVox + 1
    
    rows = mask.shape[0]
    cols = mask.shape[1]
    slices = mask.shape[2]
    
    features = np.zeros([rows,cols,slices,featurenumbers])
    
    print "Computing"
    featureList = clf.predict(dwiTest)
    
    for nVox in range(voxelList.shape[0]):
        x = voxelList[nVox,0]
        y = voxelList[nVox,1]
        z = voxelList[nVox,2]
        features[x,y,z,:] = featureList[nVox,:]
    features = features*np.power(10,scalesint)        
    hdr = dwi_nii.header
    for feature_index in range(featurenumbers):
        feature_nii = nib.Nifti1Image(features[:,:,:,feature_index], dwi_nii.get_affine(), hdr)
        feature_name = os.path.join(directory,"SRDN_feature_" + "%02d" % feature_index + "_sub_" + "%02d" % iMask + ".nii.gz")
        feature_nii.to_filename(feature_name)
    
end = time.time()
print "Test took ", (end-start)
