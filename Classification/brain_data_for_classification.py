import sys
sys.path.insert(0, '/NIRAL/work/ugor/source/challenge-brain/Pytorch-lightning/challenge-brain/Librairies')

import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pytorch_lightning as pl 

from vtk.util.numpy_support import vtk_to_numpy
import vtk

import nibabel as nib
from fsl.data import gifti
from tqdm import tqdm
from sklearn.utils import class_weight

import utils
from utils import ReadSurf, PolyDataToTensors

import pandas as pd

class BrainIBISDataset(Dataset):
    def __init__(self,df,path_data,path_ico,transform = None,version=None, column_subject_id='Subject_ID', column_age='Age', column_hemisphere='Hemisphere'):
        self.df = df
        self.path_data = path_data
        self.path_ico = path_ico
        self.transform = transform
        self.version = version
        self.column_subject_id =column_subject_id
        self.column_age = column_age
        self.column_hemisphere=column_hemisphere

    def __len__(self):
        return(len(self.df)) 

    def __getitem__(self,idx):

        #Load Data
        row = self.df.loc[idx]
        number_brain = row[self.column_subject_id]

        hemishpere = row[self.column_hemisphere]

        l_version = ['V06','V12']
        idx_version = int(row[self.column_age])
        version = l_version[idx_version]

        l_features = []

        path_eacsf = f"{self.path_data}/{number_brain}/{version}/eacsf/{hemishpere}_eacsf.txt"
        path_sa =    f"{self.path_data}/{number_brain}/{version}/sa/{hemishpere}_sa.txt"
        path_thickness = f"{self.path_data}/{number_brain}/{version}/thickness/{hemishpere}_thickness.txt"

        eacsf = open(path_eacsf,"r").read().splitlines()
        eacsf = torch.tensor([float(ele) for ele in eacsf])
        l_features.append(eacsf.unsqueeze(dim=1))

        sa = open(path_sa,"r").read().splitlines()
        sa = torch.tensor([float(ele) for ele in sa])
        l_features.append(sa.unsqueeze(dim=1))

        thickness = open(path_thickness,"r").read().splitlines()
        thickness = torch.tensor([float(ele) for ele in thickness])
        l_features.append(thickness.unsqueeze(dim=1))

        vertex_features = torch.cat(l_features,dim=1)

        Y = torch.tensor([idx_version])

        #Load  Icosahedron
        reader = utils.ReadSurf(self.path_ico)
        verts, faces, edges = utils.PolyDataToTensors(reader)
        nb_faces = len(faces)

        #Transformations 
        if self.transform:        
            verts = self.transform(verts)

        #Face Features
        faces_pid0 = faces[:,0:1]         
    
        offset = torch.zeros((nb_faces,vertex_features.shape[1]), dtype=int) + torch.Tensor([i for i in range(vertex_features.shape[1])]).to(torch.int64)
        faces_pid0_offset = offset + torch.multiply(faces_pid0, vertex_features.shape[1])      
        
        face_features = torch.take(vertex_features,faces_pid0_offset)
        
        return verts, faces,vertex_features,face_features, Y

class BrainIBISDataModule(pl.LightningDataModule):
    def __init__(self,batch_size,path_data,train_path,val_path,test_path,path_ico,train_transform=None,val_and_test_transform=None, num_workers=6, pin_memory=True, persistent_workers=True):
        super().__init__()
        self.batch_size = batch_size
        self.path_data = path_data
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.path_ico = path_ico
        self.train_transform = train_transform
        self.val_and_test_transform = val_and_test_transform
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers=persistent_workers
        
        self.nbr_features = 3


    def setup(self,stage=None):
        df_train = pd.read_csv(self.train_path)
        df_val = pd.read_csv(self.val_path)
        df_test = pd.read_csv(self.test_path)

        # Assign train/val datasets for use in dataloaders
        self.train_dataset = BrainIBISDataset(df_train,self.path_data,self.path_ico,self.train_transform)
        self.val_dataset = BrainIBISDataset(df_val,self.path_data,self.path_ico,self.val_and_test_transform)
        self.test_dataset = BrainIBISDataset(df_test,self.path_data,self.path_ico,self.val_and_test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)        

    def test_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

    def get_features(self):
        return self.nbr_features