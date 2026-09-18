#!/usr/bin/env python
# coding: utf-8

import time 
start_time = time.time()
import json
import random

from tqdm import tqdm
import numpy as np
import pandas as pd
import h5py
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
from scipy.stats import chisquare
from scipy.stats import norm

import optuna

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from copy import deepcopy
import os

### use the variables below to customize training runs ###

network = 'combined_full' # cnn, deepsets, combined_solo (DLHD no pre-training), combined_full (DLHD with pretraining)
im_type = '0.250ks' # h1, 0.250ks, 1ks, 10ks
infer_param = 'f_gas_R' # m500c, m200c, f_gas_R, f_gas_2R, f_gas_0.5R, f_gas_0.7R
band = '0.5-2keV' # a string used only in the names of stored variables
num_trials = 30 # how many trials you want optuna to optimize for
N_train = 100 # number of training epochs per phase of training

x_names = ['x','y','velocity','log_mass','sfr'] # information you want deepsets to access while training: x, y, velocty, log_mass, sfr

predicted_path = 'preds/preds_' # predicted parameter & uncertainty storage location
true_path = 'ys/ys_' # true parameter values storage location
root_storage = '' # Optuna database storage location

############### Optuna Database Setup ###############
study_name = network+'_'+infer_param+'_'+im_type+'_'+band
storage_name_pt = root_storage+study_name+".pt"
root_storage_db = 'sqlite:///'+root_storage
storage_name_db = root_storage_db+study_name+".db"




############### Reading in Data and Organizing ###############

### reading in images 

images_file = None
images = None
if im_type == '0.250ks':
    images_file = 'Maps_0.5_2.0keV_ccd_TNG300_z=0.035.0.25ksec.npy' #0.250ks
    images = np.load('data/'+images_file)
    images = np.log(images)
elif im_type == '1ks':
    images_file = 'Maps_0.5_2.0keV_ccd_TNG300_z=0.035.1.0ksec.npy' #1ks
    images = np.load('data/'+images_file)
    images = np.log(images)
elif im_type == '10ks':
    images_file = 'Maps_0.5_2.0keV_ccd_TNG300_z=0.035.10.0ksec.npy' #10ks
    images = np.load('data/'+images_file)
    images = np.log(images)
else:
    raise ValueError("im_type must be one of 0.250ks,1ks,10ks")

### read in corresponding data table. This data is initially in a different order than              
### the other catalog and the images, and only contains one axis, so we also need to do some reorganizing 

file = 'data/Params_Groups_TNG300_z=0.00.ascii_ID_fgas'
names = ['Id','m500c','fgas_R500c','a_form', 'c_200c', 'shape', 'Gamma_R500c','Gamma_vir', 'log(T)', 
           'log(Z)','[O/Fe]', '[Ne/Fe]', '[C/O]', 'axis placeholder', 'lM500c_2', 'm200c', 'f_gas_0.5R',
           'f_gas_0.7R','f_gas_R', 'f_gas_2R']
Dataset_overlord = pd.read_csv(file, names = names, delimiter = "\s+")


### Images come in 3 axes (xyz), need to make the dataframe match this info. Just duplicate the original  
### dataframe 3x. Also adding a new id for each image that gives the x,y, or z axis                      
 
DO_new = pd.DataFrame(np.repeat(Dataset_overlord.values,3,axis=0))
DO_new.columns = Dataset_overlord.columns
TM500 = DO_new['m500c']
ids = DO_new['Id']
ids_new = []
DO_new
for i in range(0,len(TM500),3):
    ids_new.append(f"{int(ids[i])}_x")
    ids_new.append(f"{int(ids[i])}_y")
    ids_new.append(f"{int(ids[i])}_z")
DO_new['Id'] = ids_new


### This step is in part a sanity check to make sure we are organizing things correctly, and also is used to sort the images 
### in the correct order. We make a mock dataframe that contains the mass and id for 
### each cluster in the image-matched data catalog and sort it by id. The mass info lets us compare with the cluster catalog to 
### check that indeed they line up, and the id will be used to sort the images in the same way. 
columns_to_grab = DO_new[['m500c','Id']]
images_bk_df = pd.DataFrame(columns_to_grab)
mock_ids = np.arange(0,10431,1)
images_bk_df['tracking_id'] = mock_ids
images_bk_df = images_bk_df.sort_values(by='Id').reset_index(drop=True)


### sorting the DO_new dataframe to match the merged_cluster_info dataframe 
DO_new = DO_new.sort_values(by='Id').reset_index(drop=True)
CNN_ids = DO_new['Id']

### using the mock dataframe to sort the images similar to the other dataframes
sorted_images = []
for i in range(len(images_bk_df['Id'])):
    index_to_grab = images_bk_df['tracking_id'][i] 
    image_to_grab = images[index_to_grab]
    sorted_images.append(image_to_grab)


### reading in deep sets data catalogs

### which simulation file to process
catalog_file = "data/TNG300.ID0_30000.gal_catalogue.expanded.hdf5" 

### which fields to store for each galaxy
columns = ["Coords_x", "Coords_y", "ID", "M500c", "MStar", "SFR", "Vels", "axis"]

merged_cluster_info = {"cluster_id": [], "m500c": [], "axis": []}
merged_galaxy_info = {
    "cluster_id": [],
    "index_in_cluster": [],
    "x": [],
    "y": [],
    "mass": [],
    "velocity": [],
    "sfr": [],
}

with h5py.File(catalog_file, "r") as hf:
    clusters = np.array(list(hf.keys()))

    print("Groups:", len(clusters))
    for cluster_id in tqdm(clusters):
        group = hf[cluster_id] 
        if "M500c" in group.keys():

    

            group = hf[cluster_id]

            x = group["Coords_x"][:]
            y = group["Coords_y"][:]
            mass = group["MStar"][:]
            velocity = group["Vels"][:]
            sfr = group["SFR"][:]
        
         
            num_stars = x.shape[0]

            assert all(elem.shape[0] == num_stars for elem in [y, mass, velocity, sfr])

            ### store galaxy fields
            merged_galaxy_info["cluster_id"] += [cluster_id] * num_stars
            merged_galaxy_info["index_in_cluster"] += list(range(num_stars))
            merged_galaxy_info["x"] += x.tolist()
            merged_galaxy_info["y"] += y.tolist()
            merged_galaxy_info["mass"] += mass.tolist()
            merged_galaxy_info["velocity"] += velocity.tolist()
            merged_galaxy_info["sfr"] += sfr.tolist()
            
            ### store cluster fields
            merged_cluster_info["cluster_id"].append(cluster_id)
            merged_cluster_info["m500c"].append(group["M500c"][()])
            merged_cluster_info["axis"].append(group["axis"][()].decode("utf-8"))

merged_galaxy_info = pd.DataFrame(merged_galaxy_info)
merged_cluster_info = pd.DataFrame(merged_cluster_info)


### clean up everything and make some conveniently normalized fields
merged_galaxy_info["log_mass"] = np.log10(merged_galaxy_info["mass"])
merged_galaxy_info["velocity/1000"] = merged_galaxy_info["velocity"] / 1000
merged_galaxy_info["R"] = np.sqrt(merged_galaxy_info["x"]**2+merged_galaxy_info["y"]**2)
merged_cluster_info["cluster_unique_id"] = merged_cluster_info["cluster_id"].apply(lambda x: x.split("_")[0])

merged_galaxy_info["cluster_unique_id"] = (
    merged_galaxy_info["cluster_id"].str.split("_").str[0]
)
merged_galaxy_info["galaxy_unique_id"] = (
    merged_galaxy_info["cluster_unique_id"]
    + "-"
    + merged_galaxy_info["index_in_cluster"].astype(str)
)

merged_cluster_info['f_gas_R'] = DO_new['f_gas_R']
merged_cluster_info['f_gas_2R'] = DO_new['f_gas_2R']
merged_cluster_info['f_gas_0.5R'] = DO_new['f_gas_0.5R']
merged_cluster_info['f_gas_0.7R'] = DO_new['f_gas_0.7R']



### The galaxy table is stored on aggregate, so it will be useful 
### to define a pandas GroupBy that will allow us to index the
### data of a specific cluster
galaxies_by_cluster = merged_galaxy_info.groupby("cluster_id")
cluster_ids = list(galaxies_by_cluster.groups.keys())



############### Data Cuts ###############

# # Masking Lower Mass Galaxies

# # masking and then re-indexing, causes issues along the way
# mass_mask = merged_galaxy_info['mass'] > 10**(11)
# merged_galaxy_info = merged_galaxy_info[mass_mask]
# # Apply the mask and reset indices
# merged_galaxy_info = merged_galaxy_info[mass_mask].reset_index(drop=True)

# # carry over the ids to the cluster data frame
# merged_cluster_info = merged_cluster_info[merged_cluster_info['cluster_id'].isin(merged_galaxy_info['cluster_id'])]

# # Update galaxy_unique_id column
# merged_galaxy_info["cluster_unique_id"] = (
#     merged_galaxy_info["cluster_id"].str.split("_").str[0]
# )
# merged_galaxy_info["galaxy_unique_id"] = (
#     merged_galaxy_info["cluster_unique_id"]
#     + "-"
#     + merged_galaxy_info.groupby("cluster_unique_id").cumcount().astype(str)
# ) 


# # Adding Velocity Noise
# # Group galaxies by cluster_id
# clusters = merged_galaxy_info.groupby('cluster_unique_id')

# # Iterate over each Cluster 
# for cluster_unique_id, cluster_galaxies in clusters: 
#     # Generate a single random value for velocity noise
#     velocity_noise_value = np.random.normal(0, 1000)
    
#     # Number of galaxies in that cluster
#     num_galaxies = len(cluster_galaxies)
    
#     # Make array the size of num_galaxies with those noise values
#     velocity_noise = np.full(num_galaxies, velocity_noise_value)
    
#     # Add the same noise to each galaxy in the cluster
#     merged_galaxy_info.loc[cluster_galaxies.index, 'velocity'] += velocity_noise




#  # Adding Position Noise 
# clusters = merged_galaxy_info.groupby("cluster_unique_id")
# # Iterate over each cluster
# for cluster_unique_id, cluster_galaxies in clusters: 
#     # Generate a single random value for x and y noise
#     x_noise_value = np.random.normal(0, 500)
#     y_noise_value = np.random.normal(0, 500)
    
#     # Number of galaxies in that cluster
#     num_galaxies = len(cluster_galaxies)
    
#     # Make arrays the size of num_galaxies with those noise values
#     x_noise = np.full(num_galaxies, x_noise_value)
#     y_noise = np.full(num_galaxies, y_noise_value)
    
#     # Add the same noise to each galaxy in the cluster
#     merged_galaxy_info.loc[cluster_galaxies.index, 'x'] += x_noise
#     merged_galaxy_info.loc[cluster_galaxies.index, 'y'] += y_noise



############### Train/Test Split Setup ###############


### pick 80% of x axes ids 
tags_x = merged_cluster_info['cluster_id'][::3]
train_tags_x = tags_x.sample(frac=0.8)

### make same list but for y and z axes
train_tags_y = train_tags_x.str.replace('_x', '_y')
train_tags_z = train_tags_x.str.replace('_x', '_z')

### concatenate all the tags into a single array
train_tags = np.concatenate([train_tags_x.values, train_tags_y.values, train_tags_z.values])
np.random.shuffle(train_tags)


### make test tags in the same way
test_tags_x = tags_x[~tags_x.isin(train_tags_x)]
test_tags_y = test_tags_x.str.replace('_x', '_y')
test_tags_z = test_tags_x.str.replace('_x', '_z')

### concatenate all the test tags into a single array
test_tags = np.concatenate([test_tags_x.values, test_tags_y.values, test_tags_z.values])
np.random.shuffle(test_tags)


clid_train_CNN = train_tags
clid_test_CNN = test_tags

### creating a random 80-20 train/test split of the image data, and saving the indices of each
### note: this isn't shuffling the order of the actual data, just the array of indices of the clusters
rand_start = np.arange(images.shape[0])
np.random.shuffle(rand_start)
sorted_images = np.array(sorted_images)



############### Dictionary Setups ###############

merged_x_CNN = {}
for i in range(images.shape[0]):
    merged_x_CNN[CNN_ids[i]] = torch.FloatTensor(
        sorted_images[i]
    ).unsqueeze(0)
merged_y_CNN = dict(
    zip(
        CNN_ids,
        merged_cluster_info[infer_param].apply(
            lambda x: torch.FloatTensor([x])
        ),
    )
)



 
y_name = infer_param 
num_features = len(x_names)
merged_x = {}
for cluster_id, cluster_id_str in zip(merged_cluster_info['cluster_id'],cluster_ids): #tqdm
    
    cluster_galaxies = galaxies_by_cluster.get_group(cluster_id_str) 
    merged_x[cluster_id] = torch.FloatTensor(
        cluster_galaxies[x_names].to_numpy()
    )
merged_y = dict(
    zip(
        merged_cluster_info["cluster_id"],
        merged_cluster_info[y_name].apply(
            lambda x: torch.FloatTensor([x])
        ),
    )
)



############### Dataloader Setup ###############

class Clusters_combo(Dataset):
    def __init__(self,x,y, images, CNN_ids):
        self.cluster_ids = CNN_ids
        self.total_samples = len(self.cluster_ids)
        self.x = x
        self.y = y
        self.images = images
        
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int):
        cluster_id = self.cluster_ids[idx]
        x = self.x[cluster_id]
        y = self.y[cluster_id]
        images = self.images[cluster_id]
        return x, y, images, x.shape[0], cluster_ids
    
    def print_ids(self):
        for idx, cluster_id in enumerate(self.cluster_ids):
            print(f"Cluster ID for sample {idx}: {cluster_id}")

def collate_fn_combo(batch):
    x, y, images, sizes, batch_ids = zip(*batch)
    x_batched = torch.concat(x)
    y_batched = torch.concat(y)
    images_batched = torch.concat(images)
    return x_batched, y_batched, images_batched, sizes, batch_ids

def build_loader_combo(clids, **kwargs):
    dataset = Clusters_combo(merged_x, merged_y, merged_x_CNN, clids)
    loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn_combo,
                        **kwargs)
    return loader


test_loader_combo = build_loader_combo(clids = clid_test_CNN)

### incase you want to run tests on the entire dataset and not a specific split
all_clid = merged_cluster_info['cluster_id']
all_loader = build_loader_combo(all_clid, shuffle = False)


############### Neural Network Architectures ###############

### Deepset Network Architecture

class ClustersPhi(nn.Module):
    """Operate on galaxy features."""
    def __init__(self, num_features):
        super().__init__()

        self.fc1 = nn.Linear(num_features, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32,32)
        self.fc4 = nn.Linear(32, 32)

    def forward(self, x):

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))

        return x

    
class ClustersRho(nn.Module):
    """Operate on cluster features"""
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32,32)
        self.fc3 = nn.Linear(32, 16) 
        self.fc4 = nn.Linear(16, 2) # this model outputs mean and log-sigma

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x
    
    def forward_2(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
        

    
class DeepSet(nn.Module):
    def __init__(self, phi: nn.Module, rho: nn.Module):
        super().__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, x, sizes):
        ### preprocess on a galaxy-by-galaxy basis
        embed = self.phi.forward(x)

        ### aggregate
        embed_split = torch.tensor_split(embed, tuple(np.cumsum(sizes)[:-1]))
        sums = [torch.sum(e, dim=0, keepdim=True) for e in embed_split]
        sums = torch.cat(sums)

        ### postprocess on a cluster-by-cluster basis
        out = self.rho.forward(torch.FloatTensor(sums))

        return out
    
    def forward_2(self,x,sizes):
      
        ### preprocess on a galaxy-by-galaxy basis
        embed = self.phi.forward(x)

        ### aggregate
        embed_split = torch.tensor_split(embed, tuple(np.cumsum(sizes)[:-1]))
        sums = [torch.sum(e, dim=0, keepdim=True) for e in embed_split]
        sums = torch.cat(sums)

        ### postprocess on a cluster-by-cluster basis
        out = self.rho.forward_2(torch.FloatTensor(sums))

        return out
        



### CNN Architecture
        
hidden = 1 # number of variables trying to infer
num_channels = 1


class CNN(nn.Module):
    """Regress over image features."""
    def __init__(self):
        super().__init__()
        dr = .2 # dropout rate
        
        #input 1x128x128 ---------> output: 2*hiddenx128x128
        self.conv01 = nn.Conv2d(num_channels, 2*hidden, kernel_size=3, stride=1, padding=1,
                               padding_mode='zeros',bias=True)
        self.conv02 = nn.Conv2d(2*hidden, 2*hidden, kernel_size=3, stride=1, padding=1,
                               padding_mode='zeros', bias=True)
        self.conv03 = nn.Conv2d(2*hidden, 2*hidden, kernel_size=2, stride=2, padding=0,
                               padding_mode='zeros', bias=True)
        
        self.Bat01 = nn.BatchNorm2d(2*hidden)
        self.Bat02 = nn.BatchNorm2d(2*hidden)
        self.Bat03 = nn.BatchNorm2d(2*hidden)
        
        #input 2*hiddenx128x128 ------> output: 4*hiddenx64x64
        self.conv11 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                                padding_mode='zeros', bias=True)
        self.conv12 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                                padding_mode='zeros', bias=True)
        self.conv13 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=2, stride=2, padding=0,
                                padding_mode='zeros', bias=True)
        
        self.Bat11 = nn.BatchNorm2d(4*hidden)
        self.Bat12 = nn.BatchNorm2d(4*hidden)
        self.Bat13 = nn.BatchNorm2d(4*hidden)
        
        #input 4*hidden*64*64 -------> output: 8*hiddenx32x32
        self.conv21 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                                padding_mode='zeros', bias=True)
        self.conv22 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                                padding_mode='zeros', bias=True)
        self.conv23 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=2, stride=2, padding=0,
                                padding_mode='zeros', bias=True)
        
        self.Bat21 = nn.BatchNorm2d(8*hidden)
        self.Bat22 = nn.BatchNorm2d(8*hidden)
        self.Bat23 = nn.BatchNorm2d(8*hidden)
        
        #input 8*hiddenx32x32 ---------> output: 16*hiddenx16x16
        self.conv31 = nn.Conv2d(8*hidden, 16*hidden, kernel_size=3, stride=1, padding=1,
                                padding_mode='zeros', bias=True)
        self.conv32 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=3, stride=1, padding=1,
                                padding_mode='zeros', bias=True)
        self.conv33 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=2, stride=2, padding=0,
                                padding_mode='zeros', bias=True)
        
        self.Bat31 = nn.BatchNorm2d(16*hidden)
        self.Bat32 = nn.BatchNorm2d(16*hidden)
        self.Bat33 = nn.BatchNorm2d(16*hidden)
        
        #input 16*hiddenx16x16 ---------> output: 32*hiddenx8x8
        self.conv41 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                                padding_mode='zeros', bias=True)
        self.conv42 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                                padding_mode='zeros', bias=True)
        self.conv43 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=2, stride=2, padding=0,
                                padding_mode='zeros', bias=True)
        
        self.Bat41 = nn.BatchNorm2d(32*hidden)
        self.Bat42 = nn.BatchNorm2d(32*hidden)
        self.Bat43 = nn.BatchNorm2d(32*hidden)
        
        #input 32*hiddenx8x8 -------> output: 64*hiddenx4x4
        self.conv51 = nn.Conv2d(32*hidden, 64*hidden, kernel_size=3, stride=1, padding=1,
                                padding_mode='zeros', bias=True)
        self.conv52 = nn.Conv2d(64*hidden, 64*hidden, kernel_size=3, stride=1, padding=1,
                                padding_mode='zeros', bias=True)
        self.conv53 = nn.Conv2d(64*hidden, 64*hidden, kernel_size=2, stride=2, padding=0,
                                padding_mode='zeros', bias=True)
        
        self.Bat51 = nn.BatchNorm2d(64*hidden)
        self.Bat52 = nn.BatchNorm2d(64*hidden)
        self.Bat53 = nn.BatchNorm2d(64*hidden)
        
        #input 64*hiddenx4x4 ----------> output: 128*hiddenx1x1
        self.conv61 = nn.Conv2d(64*hidden, 128*hidden, kernel_size=2, stride=1, padding=0,
                                padding_mode='zeros', bias=True)
        self.conv62 = nn.Conv2d(128*hidden, 128*hidden, kernel_size=2, stride=1, padding=0,
                                padding_mode='zeros', bias=True)
        self.conv63 = nn.Conv2d(128*hidden, 128*hidden, kernel_size=2, stride=1, padding=0,
                                padding_mode='zeros', bias=True)
        self.Bat61 = nn.BatchNorm2d(128*hidden)
        self.Bat62 = nn.BatchNorm2d(128*hidden)
        self.Bat63 = nn.BatchNorm2d(128*hidden)
        
        self.P0 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
        self.FC1 = nn.Linear(128*hidden, 64*hidden)
        self.FC2 = nn.Linear(64*hidden, hidden*2) # returns mean and log-sigma
        self.FC3 = nn.Linear(64*hidden, hidden*16)
        
        self.dropout = nn.Dropout(p=dr)
        self.ReLU = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, image):
        
        # LAYER 1
        x = self.LeakyReLU(self.conv01(image))
        x = self.LeakyReLU(self.Bat02(self.conv02(x)))
        
        # LAYER 2
        x = self.LeakyReLU(self.Bat11(self.conv11(x)))
        x = self.LeakyReLU(self.Bat12(self.conv12(x)))
        x = self.LeakyReLU(self.Bat13(self.conv13(x)))
        
        #LAYER 3
        x = self.LeakyReLU(self.Bat21(self.conv21(x)))
        x = self.LeakyReLU(self.Bat22(self.conv22(x)))
        x = self.LeakyReLU(self.Bat23(self.conv23(x)))
        
        #LAYER 4
        x = self.LeakyReLU(self.Bat31(self.conv31(x)))
        x = self.LeakyReLU(self.Bat32(self.conv32(x)))
        x = self.LeakyReLU(self.Bat33(self.conv33(x)))
        
        #LAYER 5
        x = self.LeakyReLU(self.Bat41(self.conv41(x)))
        x = self.LeakyReLU(self.Bat42(self.conv42(x)))
        x = self.LeakyReLU(self.Bat43(self.conv43(x)))
        
        #LAYER 6
        x = self.LeakyReLU(self.Bat51(self.conv51(x)))
        x = self.LeakyReLU(self.Bat52(self.conv52(x)))
        x = self.LeakyReLU(self.Bat53(self.conv53(x)))
        
        #FINAL
        x = self.LeakyReLU(self.Bat61(self.conv61(x)))
        x = self.LeakyReLU(self.Bat62(self.conv62(x)))
        x = self.LeakyReLU(self.Bat63(self.conv63(x)))
        
        x = x.view(image.shape[0],-1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.FC3(x)

        return x

    def forward_2(self, image):
        
        # LAYER 1
        x = self.LeakyReLU(self.conv01(image))
        x = self.LeakyReLU(self.Bat02(self.conv02(x)))
        
        # LAYER 2
        x = self.LeakyReLU(self.Bat11(self.conv11(x)))
        x = self.LeakyReLU(self.Bat12(self.conv12(x)))
        x = self.LeakyReLU(self.Bat13(self.conv13(x)))
        
        #LAYER 3
        x = self.LeakyReLU(self.Bat21(self.conv21(x)))
        x = self.LeakyReLU(self.Bat22(self.conv22(x)))
        x = self.LeakyReLU(self.Bat23(self.conv23(x)))
        
        #LAYER 4
        x = self.LeakyReLU(self.Bat31(self.conv31(x)))
        x = self.LeakyReLU(self.Bat32(self.conv32(x)))
        x = self.LeakyReLU(self.Bat33(self.conv33(x)))
        
        #LAYER 5
        x = self.LeakyReLU(self.Bat41(self.conv41(x)))
        x = self.LeakyReLU(self.Bat42(self.conv42(x)))
        x = self.LeakyReLU(self.Bat43(self.conv43(x)))
        
        #LAYER 6
        x = self.LeakyReLU(self.Bat51(self.conv51(x)))
        x = self.LeakyReLU(self.Bat52(self.conv52(x)))
        x = self.LeakyReLU(self.Bat53(self.conv53(x)))
        
        #FINAL
        x = self.LeakyReLU(self.Bat61(self.conv61(x)))
        x = self.LeakyReLU(self.Bat62(self.conv62(x)))
        x = self.LeakyReLU(self.Bat63(self.conv63(x)))
        
        x = x.view(image.shape[0],-1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.FC2(x)

        return x

    
### DLHD Architecture    

class DLHD(nn.Module):
    """A model to combine features from CNN and Deep Sets."""
    def __init__(self, deepset: nn.Module, cnn: nn.Module):
        super().__init__()
        self.deepset = deepset 
        self.cnn = cnn

        self.fc1 = nn.Linear(16,16)
        self.fc2 = nn.Linear(16, 2) # returns mean and log-sigma


    def forward(self, x, sizes, image):
            ### preprocess galaxy data
            deepset_out =  self.deepset.forward(x, sizes)
            # preprocess images
            cnn_out = self.cnn.forward(image)

            ### Initialize a list to store dot products
            dot_products = []

            ### Loop through each pair of tensors in the batch
            for ds_tensor, cnn_tensor in zip(deepset_out, cnn_out):
                dot_product = torch.mul(ds_tensor, cnn_tensor)
                dot_products.append(dot_product)
            ### Concatenate the results to form a tensor
            out = torch.stack(dot_products)

            ### postprocess
            out = torch.relu(self.fc1(out)) 
            out = self.fc2(out)

            return out



minimum = np.min(merged_cluster_info[infer_param]) 
maximum = np.max(merged_cluster_info[infer_param]) 

checkpoint_array = []


############### Training DLHD ###############

def criterion1(pred, y):
    return F.mse_loss(pred[:,0], y, reduction="mean")
epsilon = 1e-5
def criterion2(pred, y):
    return -(-pred[:,1] - ((y-(pred[:,0]))/(torch.exp(pred[:,1])+epsilon))**2/2).sum() 
# def criterion3(pred,y):
#     loss1 = torch.mean((pred[:,0]-y)**2,axis=0)
#     loss2 = torch.mean(((pred[:,0]-y)**2 - pred[:,1]**2)**2,axis=0)
#     loss  = torch.mean(torch.log(loss1) + torch.log(loss2)) 
    return loss  


best_scatter_global = float('inf')

if os.path.exists(storage_name_pt):
    prev_checkpoint = torch.load(storage_name_pt,weights_only=False)
    best_scatter_global = prev_checkpoint['scatter']
    print(f"Resuming from previous best model with scatter {best_scatter_global:.6f}")

def objective(trial):
    global best_scatter_global

    phi = ClustersPhi(num_features)
    rho = ClustersRho()
    deepset = DeepSet(phi,rho)
    cnn = CNN()
    model = DLHD(deepset, cnn)

    optimizer = optim.Adam(model.parameters(), lr=trial.suggest_float('lr',1e-5,1e-3), weight_decay=trial.suggest_float('wd',1e-5,1e-3))

    inner_base_ids = np.array(sorted(set(t.rsplit('_', 1)[0] for t in clid_train_CNN)))
    np.random.shuffle(inner_base_ids)
    n_inner_train = int(len(inner_base_ids) * 0.8)
    inner_train_set = set(inner_base_ids[:n_inner_train])
    inner_valid_set = set(inner_base_ids[n_inner_train:])

    clid_inner_train = [t for t in clid_train_CNN if t.rsplit('_', 1)[0] in inner_train_set]
    clid_inner_valid = [t for t in clid_train_CNN if t.rsplit('_', 1)[0] in inner_valid_set]
    random.shuffle(clid_inner_train)
    random.shuffle(clid_inner_valid)

    train_loader = build_loader_combo(clids=clid_inner_train)
    valid_loader = build_loader_combo(clids=clid_inner_valid)
    Ntrain = len(train_loader.dataset)
    Nvalid = len(valid_loader.dataset)

    trloss_rec = []
    teloss_rec = []
    def train_epoch_combined(model,optimizer,criterion):
        trloss = 0 
        for x, y, images, sizes, _ in tqdm(train_loader, disable=True): 
            optimizer.zero_grad()
            images = images.unsqueeze(1)
            pred = model.forward(x, sizes, images) 
            y = (y-minimum)/(maximum - minimum) #normalize values to be between 0 and 1 for convenience
            loss = criterion(pred, y) 

            loss.backward()
            optimizer.step()

            trloss += loss.detach().flatten().numpy()[0]

        teloss = 0
        with torch.no_grad():
            for x, y, images, sizes, _ in tqdm(valid_loader, disable=True):
                images = images.unsqueeze(1)
                pred = model.forward(x, sizes, images) 
                y = (y-minimum)/(maximum - minimum)
                loss = criterion(pred, y) 

                teloss += loss.detach().flatten().numpy()[0]

        trloss /= Ntrain
        teloss /= Nvalid
        trloss_rec.append(trloss)
        teloss_rec.append(teloss)
        if (i+1) % 1 == 0:
            print(f" epoch: {i+1}/{N_epochs}; train loss: {trloss:.6f}; valid loss: {teloss:.6f}")



    # defining new train functions to train networks independently
    def train_epoch_ds(model,optimizer,criterion):
        trloss = 0 
        for x, y, images, sizes, _ in tqdm(train_loader, disable=True): 
            optimizer.zero_grad()
            pred = model.deepset.forward_2(x, sizes)
            y = (y-minimum)/(maximum - minimum)
            loss = criterion(pred, y) 

            loss.backward()
            optimizer.step()

            trloss += loss.detach().flatten().numpy()[0]

        teloss = 0
        with torch.no_grad():
            for x, y, images, sizes, _ in tqdm(valid_loader,disable=True): 
                pred = model.deepset.forward_2(x, sizes)
                y = (y-minimum)/(maximum - minimum)
                loss = criterion(pred, y) 

                teloss += loss.detach().flatten().numpy()[0]

        trloss /= Ntrain
        teloss /= Nvalid
        trloss_rec.append(trloss)
        teloss_rec.append(teloss)
        if (i+1) % 1 == 0:
            print(f" epoch: {i+1}/{N_epochs}; train loss: {trloss:.6f}; valid loss:{teloss:.6f}")    

    def train_epoch_cnn(model,optimizer,criterion):
        trloss = 0 
        for x, y, images, sizes, _ in tqdm(train_loader,disable=True): 
            model.train()
            optimizer.zero_grad()
            images = images.unsqueeze(1)
            pred = model.cnn.forward_2(images)
            y = (y-minimum)/(maximum - minimum)
            loss = criterion(pred, y) 



            loss.backward()
            optimizer.step()

            trloss += loss.detach().flatten().numpy()[0]

        teloss = 0
        with torch.no_grad():
            for x, y, images, sizes, _ in tqdm(valid_loader,disable=True): 
                model.eval()
                images = images.unsqueeze(1)
                pred = model.cnn.forward_2(images)
                y = (y-minimum)/(maximum - minimum)
                loss = criterion(pred, y) 

                teloss += loss.detach().flatten().numpy()[0]

        trloss /= Ntrain
        teloss /= Nvalid
        trloss_rec.append(trloss)
        teloss_rec.append(teloss)
        if (i+1) % 1 == 0:
            print(f" epoch: {i+1}/{N_epochs}; train loss: {trloss:.6f}; valid loss: {teloss:.6f}")    
  
    N_epochs = N_train
    wait = 0
    min_change = 1e-6
    valoss_min = np.inf
    # Pretrain_epochs = 0
    broken = False
    patience = 20

    best_model_weights = deepcopy(model.state_dict())
    best_ds_model_weights = deepcopy(model.deepset.state_dict())
    best_cnn_model_weights = deepcopy(model.cnn.state_dict())
    
    
    scatter = 0
    ys = []
    preds = []
    
    
    if network == 'deepsets':

        valoss_min = np.inf
        epochs_since_improve = 0
        broken = False
        print('---- DS Pretraining ----')
        for i in range(N_epochs):
            train_epoch_ds(model,optimizer,criterion1)
            current_teloss = teloss_rec[-1]
            if current_teloss < valoss_min:
                valoss_min = current_teloss
                best_ds_model_weights = deepcopy(model.deepset.state_dict())
                broken = True
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= patience:
                    print(f'DS phase 1 stopped early at epoch {i+1}/{N_epochs}')
                    break
        if(broken):
            model.deepset.load_state_dict(best_ds_model_weights) 
        # Pretrain_epochs += N_epochs

        valoss_min = np.inf
        epochs_since_improve = 0
        broken = False
        print('---- DS Training ----')
        for i in range(N_epochs):
            train_epoch_ds(model,optimizer,criterion2)
            current_teloss = teloss_rec[-1]
            if current_teloss < valoss_min:
                valoss_min = current_teloss
                best_ds_model_weights = deepcopy(model.deepset.state_dict())
                broken = True
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= patience:
                    print(f'DS phase 2 stopped early at epoch {i+1}/{N_epochs}')
                    break

        ys = []
        preds = []
        model.deepset.load_state_dict(best_ds_model_weights)
        with torch.no_grad():
            for x, y, images, sizes, _ in tqdm(test_loader_combo):
                images = images.unsqueeze(1)
                pred = model.deepset.forward_2(x,sizes)
                pred[:,0] = pred[:,0] * (maximum - minimum) + minimum # undo normalization
                pred[:,1] += np.log(maximum - minimum) # undo normalization
                ys.append(y)
                preds.append(pred)
        ys = np.concatenate(ys)
        preds = np.concatenate(preds)
        scatter = np.std(preds[:,0]-ys)
        r2 = r2_score(ys,preds[:,0])
        if scatter < best_scatter_global:
            best_scatter_global = scatter

            checkpoint = { 
                'model': deepcopy(model.state_dict()),
                'ds_model': deepcopy(model.deepset.state_dict()),
                'cnn_model': deepcopy(model.cnn.state_dict()),
                'optimizer': deepcopy(optimizer.state_dict()),
                'scatter': best_scatter_global
            }

            checkpoint_array.append(checkpoint)
            torch.save(checkpoint, storage_name_pt)
            export_preds = np.array(preds)
            export_ys = np.array(ys)
            np.save(predicted_path+network+'_'+infer_param+'.npy',export_preds)
            np.save(true_path+network+'_'+infer_param+'.npy',export_ys)

    elif network == 'cnn':

        valoss_min = np.inf
        epochs_since_improve = 0
        broken = False
        print('---- CNN Pretraining ----')
        for i in range(N_epochs):
            train_epoch_cnn(model, optimizer, criterion1)
            current_teloss = teloss_rec[-1]
            if current_teloss < valoss_min:
                valoss_min = current_teloss
                best_cnn_model_weights = deepcopy(model.cnn.state_dict())
                broken = True
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= patience:
                    print(f'Phase 1 stopped early at epoch {i+1}/{N_epochs}')
                    break

        if(broken):
            model.cnn.load_state_dict(best_cnn_model_weights)
        # Pretrain_epochs += N_epochs

        valoss_min = np.inf
        epochs_since_improve = 0
        broken = False
        print('---- CNN Training ----')
        for i in range(N_epochs):
            train_epoch_cnn(model, optimizer, criterion2)
            current_teloss = teloss_rec[-1]
            if current_teloss < valoss_min:
                valoss_min = current_teloss
                best_cnn_model_weights = deepcopy(model.cnn.state_dict())
                broken = True
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= patience:
                    print(f'Phase 2 stopped early at epoch {i+1}/{N_epochs}')
                    break

        ys = []
        preds = []
        model.cnn.load_state_dict(best_cnn_model_weights)
        with torch.no_grad():
            for x, y, images, sizes, _ in tqdm(test_loader_combo):
                images = images.unsqueeze(1)
                pred = model.cnn.forward_2(images)
                pred[:,0] = pred[:,0] * (maximum - minimum) + minimum # undo normalization 
                pred[:,1] += np.log(maximum - minimum)  # undo normalization
                ys.append(y)
                preds.append(pred)
        ys = np.concatenate(ys)
        preds = np.concatenate(preds)
        scatter = np.std(preds[:,0]-ys)
        r2 = r2_score(ys,preds[:,0])
        if scatter < best_scatter_global:
            best_scatter_global = scatter
            checkpoint = { 
                'model': deepcopy(model.state_dict()),
                'ds_model': deepcopy(model.deepset.state_dict()),
                'cnn_model': deepcopy(model.cnn.state_dict()),
                'optimizer': deepcopy(optimizer.state_dict()),
                'scatter': best_scatter_global
            }
            checkpoint_array.append(checkpoint)
            torch.save(checkpoint, storage_name_pt)
            export_preds = np.array(preds)
            export_ys = np.array(ys)
            np.save(predicted_path+network+'_'+infer_param+'_'+im_type+'_'+band+'.npy',export_preds)
            np.save(true_path+network+'_'+infer_param+'_'+im_type+'_'+band+'.npy',export_ys)

    elif network == 'combined_solo':

        valoss_min = np.inf
        epochs_since_improve = 0
        broken = False
        print('---- DLHD Pretraining ----')
        for i in range(N_epochs):
            train_epoch_combined(model,optimizer,criterion1)
            current_teloss = teloss_rec[-1]
            if current_teloss < valoss_min:
                valoss_min = current_teloss
                best_model_weights = deepcopy(model.state_dict())
                broken = True
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= patience:
                    print(f'Combined-solo phase 1 stopped early at epoch {i+1}/{N_epochs}')
                    break
        if(broken):
            model.load_state_dict(best_model_weights)
        # Pretrain_epochs += N_epochs

        valoss_min = np.inf
        epochs_since_improve = 0
        broken = False
        print('---- DLHD Training ----')
        for i in range(N_epochs):
            train_epoch_combined(model,optimizer,criterion2)
            current_teloss = teloss_rec[-1]
            if current_teloss < valoss_min:
                valoss_min = current_teloss
                best_model_weights = deepcopy(model.state_dict())
                broken = True
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= patience:
                    print(f'Combined-solo phase 2 stopped early at epoch {i+1}/{N_epochs}')
                    break

        ys = []
        preds = []
        model.load_state_dict(best_model_weights)
        with torch.no_grad():
            for x, y, images, sizes, _ in tqdm(test_loader_combo):
                images = images.unsqueeze(1)
                pred = model.forward(x, sizes, images)
                pred[:,0] = pred[:,0] * (maximum - minimum) + minimum # undo normalization
                pred[:,1] += np.log(maximum - minimum) # undo normalization
                ys.append(y)
                preds.append(pred)
        ys = np.concatenate(ys)
        preds = np.concatenate(preds)
        scatter = np.std(preds[:,0]-ys)
        r2 = r2_score(ys,preds[:,0])
        if scatter < best_scatter_global:
            best_scatter_global = scatter

            checkpoint = { 
                'model': deepcopy(model.state_dict()),
                'ds_model': deepcopy(model.deepset.state_dict()),
                'cnn_model': deepcopy(model.cnn.state_dict()),
                'optimizer': deepcopy(optimizer.state_dict()),
                'scatter': best_scatter_global
            }

            checkpoint_array.append(checkpoint)
            torch.save(checkpoint, storage_name_pt)
            export_preds = np.array(preds)
            export_ys = np.array(ys)
            np.save(predicted_path+network+'_'+infer_param+'_'+im_type+'_'+band+'.npy',export_preds)
            np.save(true_path+network+'_'+infer_param+'_'+im_type+'_'+band+'.npy',export_ys)

    elif network == 'combined_full':

        print('---- DS Pretraining ----')
        valoss_min = np.inf
        epochs_since_improve = 0
        broken = False
        for i in range(N_epochs):
            train_epoch_ds(model,optimizer,criterion1)
            current_teloss = teloss_rec[-1]
            if current_teloss < valoss_min:
                valoss_min = current_teloss
                best_ds_model_weights = deepcopy(model.deepset.state_dict())
                broken = True
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= patience:
                    print(f'DS phase 1 stopped early at epoch {i+1}/{N_epochs}')
                    break
        if(broken):
            model.deepset.load_state_dict(best_ds_model_weights)
        # Pretrain_epochs += N_epochs
        print('---- DS Training ----')
        valoss_min = np.inf
        epochs_since_improve = 0
        broken = False
        for i in range(N_epochs):
            train_epoch_ds(model,optimizer,criterion2)
            current_teloss = teloss_rec[-1]
            if current_teloss < valoss_min:
                valoss_min = current_teloss
                best_ds_model_weights = deepcopy(model.deepset.state_dict())
                broken = True
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= patience:
                    print(f'DS phase 2 stopped early at epoch {i+1}/{N_epochs}')
                    break
        # Pretrain_epochs += N_epochs

        print('---- CNN Pretraining ----')
        valoss_min = np.inf
        epochs_since_improve = 0
        broken = False
        for i in range(N_epochs):
            train_epoch_cnn(model,optimizer,criterion1)
            current_teloss = teloss_rec[-1]
            if current_teloss < valoss_min:
                valoss_min = current_teloss
                best_cnn_model_weights = deepcopy(model.cnn.state_dict())
                broken = True
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= patience:
                    print(f'CNN phase 1 stopped early at epoch {i+1}/{N_epochs}')
                    break
        if(broken):
            model.cnn.load_state_dict(best_cnn_model_weights)
        # Pretrain_epochs += N_epochs

        print('---- CNN Training ----')
        valoss_min = np.inf
        epochs_since_improve = 0
        broken = False
        for i in range(N_epochs):
            train_epoch_cnn(model,optimizer,criterion2)
            current_teloss = teloss_rec[-1]
            if current_teloss < valoss_min:
                valoss_min = current_teloss
                best_cnn_model_weights = deepcopy(model.cnn.state_dict())
                broken = True
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= patience:
                    print(f'CNN phase 2 stopped early at epoch {i+1}/{N_epochs}')
                    break
        # Pretrain_epochs += N_epochs

        print('---- DLHD Pretraining ----')
        valoss_min = np.inf
        epochs_since_improve = 0
        broken = False
        model.deepset.load_state_dict(best_ds_model_weights)
        model.cnn.load_state_dict(best_cnn_model_weights)
        for i in range(N_epochs):
            train_epoch_combined(model,optimizer,criterion1)
            current_teloss = teloss_rec[-1]
            if current_teloss < valoss_min:
                valoss_min = current_teloss
                best_model_weights = deepcopy(model.state_dict())
                broken = True
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= patience:
                    print(f'Combined phase 1 stopped early at epoch {i+1}/{N_epochs}')
                    break
        if(broken):
            model.load_state_dict(best_model_weights)
        # Pretrain_epochs += N_epochs

        print('---- DLHD Training ----')
        valoss_min = np.inf
        epochs_since_improve = 0
        broken = False
        for i in range(N_epochs):
            train_epoch_combined(model,optimizer,criterion2)
            current_teloss = teloss_rec[-1]
            if current_teloss < valoss_min:
                valoss_min = current_teloss
                best_model_weights = deepcopy(model.state_dict())
                broken = True
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= patience:
                    print(f'Combined phase 2 stopped early at epoch {i+1}/{N_epochs}')
                    break

        model.load_state_dict(best_model_weights)
        with torch.no_grad():
            for x, y, images, sizes, _ in tqdm(test_loader_combo):
                images = images.unsqueeze(1)
                pred = model.forward(x, sizes, images)
                pred[:,0] = pred[:,0] * (maximum - minimum) + minimum # undo normalization
                pred[:,1] += np.log(maximum - minimum) # undo normalization
                ys.append(y) 
                preds.append(pred)
        ys = np.concatenate(ys)
        preds = np.concatenate(preds)
        scatter = np.std(preds[:,0]-ys)
        r2 = r2_score(ys,preds[:,0])
        if scatter < best_scatter_global:
            best_scatter_global = scatter

            checkpoint = { 
                'model': deepcopy(model.state_dict()),
                'ds_model': deepcopy(model.deepset.state_dict()),
                'cnn_model': deepcopy(model.cnn.state_dict()),
                'optimizer': deepcopy(optimizer.state_dict()),
                'scatter': best_scatter_global
            }

            checkpoint_array.append(checkpoint)
            torch.save(checkpoint, storage_name_pt)
            export_preds = np.array(preds)
            export_ys = np.array(ys)
            np.save(predicted_path+network+'_'+infer_param+'_'+im_type+'_'+band+'.npy',export_preds)
            np.save(true_path+network+'_'+infer_param+'_'+im_type+'_'+band+'.npy',export_ys)

    return scatter 







study = optuna.create_study(study_name = study_name, storage=storage_name_db, directions = ['minimize'],load_if_exists=True)
study.optimize(objective, n_trials=num_trials)


print('Best trial:')
trial = study.best_trial
print('  Scatter: {}'.format(trial.values[0]))
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))                    

print(" --- %s seconds ---" % (time.time() - start_time))
print(f'Inferring {infer_param} with {network} using {im_type} images in {band} band')


