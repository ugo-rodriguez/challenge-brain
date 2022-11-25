#CUDA_VISIBLE_DEVICES=0


#RuntimeError: Early stopping conditioned on metric `val_loss` which is not available. Pass in or modify your `EarlyStopping` callback to use any of the following: ``
import sys
sys.path.insert(0, '/NIRAL/work/ugor/source/challenge-brain/Pytorch-lightning/challenge-brain/Librairies')

import numpy as np
import torch
import pytorch_lightning as pl 
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import monai
import nibabel as nib


from net_classification_ASD import BrainNet,BrainIcoNet, BrainIcoAttentionNet
from data_classification_ASD import BrainIBISDataModuleforClassificationASD
from logger_classification_ASD import BrainNetImageLogger

from transformation import RandomRotationTransform, GaussianNoisePointTransform, NormalizePointTransform, CenterSphereTransform


print("Import done")

def main():
    batch_size = 8
    image_size = 224
    noise_lvl = 0.03
    dropout_lvl = 0.2
    num_epochs = 100
    ico_lvl = 1
    radius=2
    lr = 1e-4

    mean = 0
    std = 0.01

    min_delta_early_stopping = 0.00
    patience_early_stopping = 20

    path_data = "/ASD/Autism/IBIS/Proc_Data/IBIS_sa_eacsf_thickness"
    train_path = "/NIRAL/work/ugor/source/challenge-brain/Pytorch-lightning/challenge-brain/Data/dataASD_V06_train.csv"
    val_path = "/NIRAL/work/ugor/source/challenge-brain/Pytorch-lightning/challenge-brain/Data/dataASD_V06_val.csv"
    test_path = "/NIRAL/work/ugor/source/challenge-brain/Pytorch-lightning/challenge-brain/Data/dataASD_V06_test.csv"
    path_ico = '/NIRAL/tools/atlas/Surface/Sphere_Template/sphere_f327680_v163842.vtk'

    list_train_transform = []    
    list_train_transform.append(CenterSphereTransform())
    list_train_transform.append(NormalizePointTransform())
    list_train_transform.append(RandomRotationTransform())
    list_train_transform.append(GaussianNoisePointTransform(mean,std))
    list_train_transform.append(NormalizePointTransform())

    train_transform = monai.transforms.Compose(list_train_transform)

    list_val_and_test_transform = []    
    list_val_and_test_transform.append(CenterSphereTransform())
    list_val_and_test_transform.append(NormalizePointTransform())

    val_and_test_transform = monai.transforms.Compose(list_val_and_test_transform)

    checkpoint_callback = ModelCheckpoint(
        dirpath='Checkpoint',
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )

    logger = TensorBoardLogger(save_dir="test_tensorboard", name="my_model")  

    brain_data = BrainIBISDataModuleforClassificationASD(batch_size,path_data,train_path,val_path,test_path,path_ico,train_transform = train_transform,val_and_test_transform =val_and_test_transform)
    nbr_features = brain_data.get_features()

    model = BrainNet(nbr_features,dropout_lvl,image_size,noise_lvl,ico_lvl,batch_size, radius=radius,lr=lr)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=min_delta_early_stopping, patience=patience_early_stopping, verbose=True, mode="min")

    image_logger = BrainNetImageLogger(num_features = nbr_features,mean = 0,std=noise_lvl)

    #trainer = Trainer(max_epochs=num_epochs,callbacks=[early_stop_callback])
    trainer = Trainer(logger=logger,max_epochs=num_epochs,callbacks=[early_stop_callback,checkpoint_callback,image_logger],accelerator="gpu")

    trainer.fit(model,datamodule=brain_data)

    trainer.test(model, datamodule=brain_data)

    torch.save(model.state_dict(), 'ModeleBrainClassificationIcoConvAvgPooling.pth')

    # x, PX = model.render(brain_data[4][0], brain_data[4][1])
    # model()

if __name__ == '__main__':
    main()