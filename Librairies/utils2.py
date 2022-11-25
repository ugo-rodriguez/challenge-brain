import numpy as np
import random

import torch

import nibabel as nib
from fsl.data import gifti
from sklearn.utils import class_weight

import utils 
from utils import ReadSurf,PolyDataToTensors

# def setup_data(split_path):
#     train_split_path = split_path+'/train.npy'
#     val_split_path = split_path+'/validation.npy'


#     train_split = np.load(train_split_path,allow_pickle=True)
#     val_split = np.load(val_split_path,allow_pickle=True)

#     y_train = train_split[:,2]
#     y_train = np.array(list(y_train[:]), dtype=float)

#     y_val = val_split[:,2]
#     y_val = np.array(list(y_val[:]), dtype=float)

#     hist, bin_edges = np.histogram(y_train, bins=5, range=(min(y_train-0.1),max(y_train+0.1)))

#     y_train = np.digitize(y_train, bin_edges) - 1
#     y_val = np.digitize(y_val, bin_edges) - 1

#     unique_classes = np.sort(np.unique(y_train))
#     unique_class_weights = np.array(class_weight.compute_class_weight('balanced', classes=unique_classes, y=y_train)) 

#     unique_classes_obj = {}
#     unique_classes_obj_str = {}
#     for uc, cw in zip(unique_classes, unique_class_weights):
#         unique_classes_obj[uc] = cw
#         unique_classes_obj_str[str(uc)] = cw

#     class_weights_train = []
#     for y in y_train:
#         class_weights_train.append(unique_classes_obj[y])

#     class_weights_val = []
#     for y in y_val:
#         class_weights_val.append(unique_classes_obj[y])

#     return train_split,val_split,class_weights_train,class_weights_val

# def setup_data_V2(split_path,version):
#     split_path = split_path+'/data'+version+'.txt'

#     list_patient = open(split_path,"r").read().splitlines()
#     list_patient = [patient.split(',') for patient in list_patient]

#     random.shuffle(list_patient)
#     percentage_split = 0.10
#     index_split = len(list_patient) - int(0.10*len(list_patient))+1

#     train_split = list_patient[:index_split]
#     val_split = list_patient[index_split:]

#     #For train
#     y_train = np.array(train_split)[:,1]
#     y_train = np.array(list(y_train[:]), dtype=int)

#     unique_classes = np.sort(np.unique(y_train))
#     unique_class_weights = np.array(class_weight.compute_class_weight('balanced', classes=unique_classes, y=y_train))

#     unique_classes_obj = {}
#     for uc, cw in zip(unique_classes, unique_class_weights):
#         unique_classes_obj[uc] = cw
    
#     class_weights_train = []
#     for y in y_train:
#         class_weights_train.append(unique_classes_obj[y])

#     #For val
#     y_val = np.array(val_split)[:,1]
#     y_val = np.array(list(y_val[:]), dtype=int)

#     unique_classes = np.sort(np.unique(y_val))
#     unique_class_weights = np.array(class_weight.compute_class_weight('balanced', classes=unique_classes, y=y_val))

#     unique_classes_obj = {}
#     for uc, cw in zip(unique_classes, unique_class_weights):
#         unique_classes_obj[uc] = cw
    
#     class_weights_val = []
#     for y in y_val:
#         class_weights_val.append(unique_classes_obj[y])

#     return train_split,val_split,class_weights_train,class_weights_val

# def get_features(np_split,idx):
#     item = np_split[idx]

#     idx_space = random.randint(0,1)
#     idx_feature = random.randint(0,1)
#     if idx_space == 0:
#         data_dir = '/CMF/data/geometric-deep-learning-benchmarking/Data/Regression/Template_Space'
#     else:
#         data_dir = '/CMF/data/geometric-deep-learning-benchmarking/Data/Regression/Native_Space'        
#     l_space = ['template','native']
#     l_hemishpere =['L','R']        
#     path_features = f"{data_dir}/regression_{l_space[idx_space]}_space_features/{item[0]}_{l_hemishpere[idx_feature]}.shape.gii"
    
#     #path_features = '/ASD/Autism/IBIS/Proc_Data/IBIS_sa_eacsf_thickness'

#     vertex_features = torch.from_numpy(gifti.loadGiftiVertexData(path_features)[1]) # vertex features

#     age_at_birth = item[2]

#     return vertex_features,age_at_birth

# def get_features_V2(np_split,idx,version):
#     item = np_split[idx]

#     path_eacsf = f"/ASD/Autism/IBIS/Proc_Data/IBIS_sa_eacsf_thickness/{item[0]}/{version}/eacsf/left_eacsf.txt"
#     path_sa =    f"/ASD/Autism/IBIS/Proc_Data/IBIS_sa_eacsf_thickness/{item[0]}/{version}/sa/left_sa.txt"
#     path_thickness = f"/ASD/Autism/IBIS/Proc_Data/IBIS_sa_eacsf_thickness/{item[0]}/{version}/thickness/left_thickness.txt"

#     eacsf = open(path_eacsf,"r").read().splitlines()
#     eacsf = torch.tensor([float(ele) for ele in eacsf])

#     sa = open(path_sa,"r").read().splitlines()
#     sa = torch.tensor([float(ele) for ele in sa])

#     thickness = open(path_thickness,"r").read().splitlines()
#     thickness = torch.tensor([float(ele) for ele in thickness])

#     l_features = [eacsf.unsqueeze(dim=1),sa.unsqueeze(dim=1),thickness.unsqueeze(dim=1)]
#     vertex_features = torch.cat(l_features,dim=1)

#     age_at_birth = int(item[1])

#     return vertex_features,age_at_birth

# def ReadIcoSurf(path_ico):
#     # load icosahedron
#     ico_surf = nib.load(path_ico)

#     # extract points and faces
#     verts = torch.from_numpy(ico_surf.agg_data('pointset'))
#     faces = torch.from_numpy(ico_surf.agg_data('triangle'))

#     return verts,faces

# def ReadIcoSurfV2(path_ico):
#     reader = ReadSurf(path_ico)
#     verts, faces, edges = utils.PolyDataToTensors(reader)
#     return verts,faces 

def get_neighbors(list_edges,nbr_cam):
    nbr_cam = torch.max(list_edges)
    neighbors = [[] for i in range(nbr_cam)]
    for edge in list_edges:
        v1 = edge[0]
        v2 = edge[1]
        neighbors[v1].append(v2)
        neighbors[v2].append(v1)
    return neighbors

def sort_neighbors(list_neighbors):
    nbr_cam = len(list_neighbors)
    new_neighbors = [[] for i in range(nbr_cam)]
    for i in range(nbr_cam):
        neighbors = list_neighbors[i].copy()
        vert = neighbors[0]
        new_neighbors[i].append(vert)
        neighbors.remove(vert)
        while len(neighbors) != 0:
            common_neighbors = list(set(neighbors).intersection(list_neighbors[vert]))
            vert = common_neighbors[0]
            new_neighbors[i].append(vert)
            neighbors.remove(vert)
    return new_neighbors

def get_mat_neighbors(list_neighbors):
    nbr_cam = len(list_neighbors)
    first_row = [[list_neighbors[i][0],list_neighbors[i][1],list_neighbors[i][2]] for i in range(nbr_cam)]
    second_row =  [[-1,i,-1] for i in range(nbr_cam)]
    third_row = []
    mat_neighbors = [[],[],[]]
    for i in range(nbr_cam):

        #First row 
        mat_neighbors[0].append(list_neighbors[i][0]),mat_neighbors[0].append(list_neighbors[i][1]),mat_neighbors[0].append(list_neighbors[i][2])

        #Second row
        mat_neighbors[1].append(nbr_cam),mat_neighbors[1].append(i),mat_neighbors[1].append(nbr_cam)

        #Third row
        mat_neighbors[2].append(list_neighbors[i][3]),mat_neighbors[2].append(list_neighbors[i][4]) 
        if len(list_neighbors[i]) == 5:
            mat_neighbors[1].append(nbr_cam)
        else:
            mat_neighbors[2].append(list_neighbors[i][5])
    return mat_neighbors

def IcosahedronConv(x,mat_neighbors):
    batch_size,nbr_cam,nbr_features = x.size()
    z = torch.zeros(batch_size,1,nbr_features)
    l = [x,z]
    x_with_0 = torch.cat(l,dim=1)

    m = torch.tensor([[np.array(mat_neighbors)*nbr_features+j+i*(nbr_cam+1)*nbr_features for j in range(nbr_features)] for i in range(batch_size)])

    new_x = torch.take(x_with_0,m)

    initial_size = [batch_size,nbr_features,nbr_cam]
    size_reshape = [batch_size*nbr_features,1,3,3*nbr_cam]
    x_reshape = new_x.contiguous().view(size_reshape)

    conv = nn.Conv2d(1,1,kernel_size=(3,3),stride=3,padding=0)

    output = conv(x_reshape)
    output = output.contiguous().view(initial_size)
    output = output.permute(0,2,1)




