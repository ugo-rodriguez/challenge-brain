import sys
sys.path.insert(0, '/NIRAL/work/ugor/source/challenge-brain/Pytorch-lightning/challenge-brain/Librairies')

import numpy as np
import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl 
import torchvision.models as models
from torch.nn.functional import softmax
import torchmetrics

import utils
from utils import ReadSurf, PolyDataToTensors, CreateIcosahedron

# datastructures
from pytorch3d.structures import Meshes

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRendererWithFragments, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, SoftPhongShader, AmbientLights, PointLights, TexturesUV, TexturesVertex,
)
from pytorch3d.vis.plotly_vis import plot_scene

import plotly.express as px
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import cv2

class GaussianNoise(nn.Module):
    def __init__(self, mean=0.0, std=0.01):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.std = std
        
    def forward(self, x):
        if(self.training):
            return x + torch.normal(self.mean, self.std,size=x.shape, device=x.device)*(x!=0) # add noise on sphere (not on background)
        return x


class SelfAttention(nn.Module):
    def __init__(self,in_units,out_units):
        super().__init__()


        self.W1 = nn.Linear(in_units, out_units)
        self.V = nn.Linear(out_units, 1)
        self.Tanh = nn.Tanh()
        self.Sigmoid = nn.Sigmoid()
    
    def forward(self, query, values):        
        score = self.Sigmoid(self.V(self.Tanh(self.W1(query))))
        
        attention_weights = score/torch.sum(score, dim=1,keepdim=True)

        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, score

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module
 
    def forward(self, input_seq):
        assert len(input_seq.size()) > 2

        size = input_seq.size()

        batch_size = size[0]
        time_steps = size[1]

        size_reshape = [batch_size*time_steps] + list(size[2:])
        reshaped_input = input_seq.contiguous().view(size_reshape)

        output = self.module(reshaped_input)
        
        output_size = output.size()
        output_size = [batch_size, time_steps] + list(output_size[1:])
        output = output.contiguous().view(output_size)

        return output

class MaxPoolImages(nn.Module):
    def __init__(self, nbr_images = 12):
        super().__init__()
        self.nbr_images = nbr_images
        self.max_pool = nn.MaxPool1d(self.nbr_images)
 
    def forward(self,x):
        x = x.permute(0,2,1)
        output = self.max_pool(x)
        output = output.squeeze(dim=2)

        return output

class AvgPoolImages(nn.Module):
    def __init__(self, nbr_images = 12):
        super().__init__()
        self.nbr_images = nbr_images
        self.avg_pool = nn.AvgPool1d(self.nbr_images)
 
    def forward(self,x):
        x = x.permute(0,2,1)
        output = self.avg_pool(x)
        output = output.squeeze(dim=2)

        return output

class IcosahedronConv2d(nn.Module):
    def __init__(self,module,verts,list_edges):
        super().__init__()
        self.module = module
        self.verts = verts
        self.list_edges = list_edges
        self.nbr_vert = np.max(self.list_edges)+1

        self.list_neighbors = self.get_neighbors()
        self.list_neighbors = self.sort_neighbors()
        self.list_neighbors = self.sort_rotation()
        mat_neighbors = self.get_mat_neighbors()

        self.register_buffer("mat_neighbors", mat_neighbors)


    
    def get_neighbors(self):
        neighbors = [[] for i in range(self.nbr_vert)]
        for edge in self.list_edges:
            v1 = edge[0]
            v2 = edge[1]
            neighbors[v1].append(v2)
            neighbors[v2].append(v1)
        return neighbors
    
    def sort_neighbors(self):
        new_neighbors = [[] for i in range(self.nbr_vert)]
        for i in range(self.nbr_vert):
            neighbors = self.list_neighbors[i].copy()
            vert = neighbors[0]
            new_neighbors[i].append(vert)
            neighbors.remove(vert)
            while len(neighbors) != 0:
                common_neighbors = list(set(neighbors).intersection(self.list_neighbors[vert]))
                vert = common_neighbors[0]
                new_neighbors[i].append(vert)
                neighbors.remove(vert)
        return new_neighbors
    
    def sort_rotation(self):
        new_neighbors = [[] for i in range(self.nbr_vert)]
        for i in range(self.nbr_vert):
            p0 = self.verts[i]
            p1 = self.verts[self.list_neighbors[i][0]]
            p2 = self.verts[self.list_neighbors[i][1]]
            v1 = p1 - p0
            v2 = p2 - p1
            vn = torch.cross(v1,v2)
            n = vn/torch.norm(vn)


            milieu = p1 + v2/2
            v3 = milieu - p0
            cg = p0 + 2*v3/3

            if (torch.dot(n,cg) > 1 ):
                new_neighbors[i] = self.list_neighbors[i]
            else:
                self.list_neighbors[i].reverse()
                new_neighbors[i] = self.list_neighbors[i]

        return new_neighbors
    
    def get_mat_neighbors(self):
        mat = torch.zeros(self.nbr_vert,self.nbr_vert*9)
        for index_cam in range(self.nbr_vert):
            mat[index_cam][index_cam*9] = 1
            for index_neighbor in range(len(self.list_neighbors[index_cam])):
                mat[self.list_neighbors[index_cam][index_neighbor]][index_cam*9+index_neighbor+1] = 1
        return mat

 
    def forward(self,x):
        batch_size,nbr_cam,nbr_features = x.size()

        x = x.permute(0,2,1)
        size_reshape = [batch_size*nbr_features,nbr_cam]
        x = x.contiguous().view(size_reshape)

        x = torch.mm(x,self.mat_neighbors)
        size_reshape2 = [batch_size,nbr_cam,nbr_features,3,3]
        x = x.contiguous().view(size_reshape2)
        x = x.permute(0,2,1,3,4)

        size_reshape3 = [batch_size*nbr_cam,nbr_features,3,3]
        x = x.contiguous().view(size_reshape3)

        output = self.module(x)
        output_channels = self.module.out_channels
        size_initial = [batch_size,nbr_cam,output_channels]
        output = output.contiguous().view(size_initial)

        return output





class BrainNet(pl.LightningModule):
    def __init__(self,nbr_features,dropout_lvl,image_size,noise_lvl,ico_lvl,batch_size, radius=1.35, lr=1e-4):
        super().__init__()

        self.save_hyperparameters()

        self.nbr_features = nbr_features
        self.dropout_lvl = dropout_lvl
        self.image_size = image_size
        self.noise_lvl = noise_lvl 
        self.batch_size = batch_size
        self.radius = radius

        ico_sphere = utils.CreateIcosahedron(self.radius, ico_lvl)
        ico_sphere_verts, ico_sphere_faces, self.ico_sphere_edges = utils.PolyDataToTensors(ico_sphere)
        self.ico_sphere_verts = ico_sphere_verts
        self.ico_sphere_edges = np.array(self.ico_sphere_edges)
        R=[]
        T=[]
        for coords_cam in self.ico_sphere_verts.tolist():
            camera_position = torch.FloatTensor([coords_cam])
            R_current = look_at_rotation(camera_position)
            # check if camera coords vector and up vector for R are collinear
            #if torch.equal(torch.cross(camera_position,torch.tensor([[0.,1.,0.]])),torch.tensor([[0., 0., 0.]])): 
            #    R = look_at_rotation(camera_position, up = torch.tensor([[0.0, 0.0, 1.0]]),)    
            T_current = -torch.bmm(R_current.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

            R.append(R_current)
            T.append(T_current)
        self.R=torch.cat(R)
        self.T=torch.cat(T)
        self.nbr_cam = len(self.R)

        efficient_net = models.resnet18()
        #efficient_net = models.efficientnet_b0()
        #efficient_net.features[0][0] = nn.Conv2d(self.nbr_features, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        efficient_net.fc = Identity()

        self.drop = nn.Dropout(p=self.dropout_lvl)

        self.noise = GaussianNoise(mean=0.0, std=noise_lvl)
        self.TimeDistributed = TimeDistributed(efficient_net)

        self.WV = nn.Linear(512, 256)

        # conv2dForQuery = nn.Conv2d(1280, 1280, kernel_size=(3,3),stride=2,padding=0) #1280,512
        # conv2dForValues = nn.Conv2d(512, 512, kernel_size=(3,3),stride=2,padding=0)  #512,512

        # self.IcosahedronConv2dForQuery = IcosahedronConv2d(conv2dForQuery,self.ico_sphere_verts,self.ico_sphere_edges)
        # self.IcosahedronConv2dForValues = IcosahedronConv2d(conv2dForValues,self.ico_sphere_verts,self.ico_sphere_edges)

        self.Attention = SelfAttention(512, 128)
        self.Classification = nn.Linear(256, 2)

        self.Sigmoid = nn.Sigmoid()
        self.loss = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()




        # Initialize a perspective camera.
        self.cameras = FoVPerspectiveCameras()

        
        # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
        raster_settings = RasterizationSettings(
            image_size=self.image_size, 
            blur_radius=0, 
            faces_per_pixel=1, 
            max_faces_per_bin=100000
        )
        # We can add a point light in front of the object. 

        lights = AmbientLights()
        rasterizer = MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings
            )
        self.phong_renderer = MeshRendererWithFragments(
            rasterizer=rasterizer,
            shader=HardPhongShader(cameras=self.cameras, lights=lights)
        )
 
    def forward(self, x):

        V, F, VF, FF = x

        V = V.to(self.device)
        F = F.to(self.device)
        VF = VF.to(self.device)
        FF = FF.to(self.device)

        x, PF = self.render(V,F,VF,FF)

        x = self.noise(x)
        query = self.TimeDistributed(x)

        values = self.WV(query)
        
        x_a, w_a = self.Attention(query, values)
        x_a = self.drop(x_a)
        x = self.Classification(x_a)

        return x

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def render(self,V,F,VF,FF):

        textures = TexturesVertex(verts_features=VF)
        meshes = Meshes(
            verts=V,   
            faces=F, 
            textures=textures
        )

        # To plot the sphere
        # fig = plot_scene({"subplot1":{"meshes":meshes}})
        # fig.show()

        PF = []
        #for i in range(2):  # multiple views of the object
        for i in range(self.nbr_cam): 

            pix_to_face = self.GetView(meshes,i)
            PF.append(pix_to_face.unsqueeze(dim=1))

        PF = torch.cat(PF, dim=1)
        l_features = []
        for index in range(FF.shape[-1]):
            l_features.append(torch.take(FF[:,:,index],PF)*(PF >= 0)) # take each feature
        x = torch.cat(l_features,dim=2)

        return x, PF

    def training_step(self, train_batch, batch_idx):

        V, F, VF, FF, Y = train_batch

        Y = Y.squeeze(dim=1)

        x = self((V, F, VF, FF)) 

        loss = self.loss(x,Y) 

        self.log('train_loss', loss, batch_size=self.batch_size)

        predictions = torch.argmax(x, dim=1, keepdim=True)
        self.train_accuracy(predictions.reshape(-1, 1), Y.reshape(-1, 1))
        self.log("train_acc", self.train_accuracy, batch_size=self.batch_size)  

        return loss

    def validation_step(self,val_batch,batch_idx):
        
        V, F, VF, FF, Y = val_batch

        Y = Y.squeeze(dim=1)

        x = self((V, F, VF, FF)) 

        #Last loss : for BCE lines
        # x = self.Sigmoid(x).squeeze(dim=1)
        #loss = self.loss(x,Y.to(torch.float32)) 

        loss = self.loss(x,Y) 

        self.log('val_loss', loss, batch_size=self.batch_size)

        predictions = torch.argmax(x, dim=1, keepdim=True)
        self.val_accuracy(predictions.reshape(-1, 1), Y.reshape(-1, 1))
        self.log("val_acc", self.val_accuracy, batch_size=self.batch_size) 


    def test_step(self,test_batch,batch_idx):
        
        V, F, VF, FF, Y = test_batch

        Y = Y.squeeze(dim=1)

        x = self((V, F, VF, FF)) 

        loss = self.loss(x,Y) 

        self.log('test_loss', loss, batch_size=self.batch_size)

        predictions = torch.argmax(x, dim=1, keepdim=True)

        output = [predictions,Y]

        return output

    def test_epoch_end(self,input_test):
        y_pred = []
        y_true = []
        for ele in input_test:
            y_pred += ele[0].tolist()
            y_true += ele[1].tolist()
        target_names = ['V06','V12']

        cf_matrix = confusion_matrix(y_true, y_pred)

        fig = px.imshow(cf_matrix,labels=dict(x="Predicted condition", y="Actual condition"),x=target_names,y=target_names)
        fig.update_xaxes(side="top")
        fig.write_image("images/confusion_matrix.png")
        fig.show()

        print(y_pred)
        print(y_true)
        print(classification_report(y_true, y_pred, target_names=target_names))

    def GetView(self,meshes,index):

        phong_renderer = self.phong_renderer.to(self.device)
        R = self.R[index][None].to(self.device)
        T = self.T[index][None].to(self.device)

        _, fragments = phong_renderer(meshes.clone(),R=R,T=T)
        pix_to_face = fragments.pix_to_face   
        pix_to_face = pix_to_face.permute(0,3,1,2)
        return pix_to_face








class BrainIcoNet(pl.LightningModule):
    def __init__(self,nbr_features,dropout_lvl,image_size,noise_lvl,ico_lvl,batch_size, radius=1.35, lr=1e-4):
        super().__init__()

        self.save_hyperparameters()

        self.nbr_features = nbr_features
        self.dropout_lvl = dropout_lvl
        self.image_size = image_size
        self.noise_lvl = noise_lvl 
        self.batch_size = batch_size
        self.radius = radius

        ico_sphere = utils.CreateIcosahedron(self.radius, ico_lvl)
        ico_sphere_verts, ico_sphere_faces, self.ico_sphere_edges = utils.PolyDataToTensors(ico_sphere)
        self.ico_sphere_verts = ico_sphere_verts
        self.ico_sphere_edges = np.array(self.ico_sphere_edges)
        R=[]
        T=[]
        for coords_cam in self.ico_sphere_verts.tolist():
            camera_position = torch.FloatTensor([coords_cam])
            R_current = look_at_rotation(camera_position)
            # check if camera coords vector and up vector for R are collinear
            #if torch.equal(torch.cross(camera_position,torch.tensor([[0.,1.,0.]])),torch.tensor([[0., 0., 0.]])): 
            #    R = look_at_rotation(camera_position, up = torch.tensor([[0.0, 0.0, 1.0]]),)    
            T_current = -torch.bmm(R_current.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

            R.append(R_current)
            T.append(T_current)
        self.R=torch.cat(R)
        self.T=torch.cat(T)
        self.nbr_cam = len(self.R)

        efficient_net = models.resnet18()
        efficient_net.fc = Identity()

        self.drop = nn.Dropout(p=self.dropout_lvl)

        self.noise = GaussianNoise(mean=0.0, std=noise_lvl)
        self.TimeDistributed = TimeDistributed(efficient_net)

        conv2d = nn.Conv2d(512, 256, kernel_size=(3,3),stride=2,padding=0) 
        self.IcosahedronConv2d = IcosahedronConv2d(conv2d,self.ico_sphere_verts,self.ico_sphere_edges)

        self.pooling = AvgPoolImages(nbr_images=self.nbr_cam)

        self.Classification = nn.Linear(256, 2)

        self.Sigmoid = nn.Sigmoid()
        self.loss = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()




        # Initialize a perspective camera.
        self.cameras = FoVPerspectiveCameras()

        
        # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
        raster_settings = RasterizationSettings(
            image_size=self.image_size, 
            blur_radius=0, 
            faces_per_pixel=1, 
            max_faces_per_bin=100000
        )
        # We can add a point light in front of the object. 

        lights = AmbientLights()
        rasterizer = MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings
            )
        self.phong_renderer = MeshRendererWithFragments(
            rasterizer=rasterizer,
            shader=HardPhongShader(cameras=self.cameras, lights=lights)
        )
 
    def forward(self, x):

        V, F, VF, FF = x

        V = V.to(self.device)
        F = F.to(self.device)
        VF = VF.to(self.device)
        FF = FF.to(self.device)

        x, PF = self.render(V,F,VF,FF)

        x = self.noise(x)
        x = self.TimeDistributed(x)

        x = self.IcosahedronConv2d(x)

        x_a = self.pooling(x)
        
        x_a = self.drop(x_a)
        x = self.Classification(x_a)

        return x

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def render(self,V,F,VF,FF):

        textures = TexturesVertex(verts_features=VF)
        meshes = Meshes(
            verts=V,   
            faces=F, 
            textures=textures
        )

        # To plot the sphere
        # fig = plot_scene({"subplot1":{"meshes":meshes}})
        # fig.show()

        PF = []
        #for i in range(2):  # multiple views of the object
        for i in range(self.nbr_cam): 

            pix_to_face = self.GetView(meshes,i)
            PF.append(pix_to_face.unsqueeze(dim=1))

        PF = torch.cat(PF, dim=1)
        l_features = []
        for index in range(FF.shape[-1]):
            l_features.append(torch.take(FF[:,:,index],PF)*(PF >= 0)) # take each feature
        x = torch.cat(l_features,dim=2)

        # for i in range(12):
        #     image = x[0,i]
        #     image = image.permute(1,2,0)
        #     print(image.shape)
        #     image = image.cpu().detach().numpy()
        #     cv2.imwrite('Image_features'+str(i)+'.png', image)

        return x, PF

    def training_step(self, train_batch, batch_idx):

        V, F, VF, FF, Y = train_batch

        Y = Y.squeeze(dim=1)

        x = self((V, F, VF, FF)) 

        loss = self.loss(x,Y) 

        self.log('train_loss', loss, batch_size=self.batch_size)

        predictions = torch.argmax(x, dim=1, keepdim=True)
        self.train_accuracy(predictions.reshape(-1, 1), Y.reshape(-1, 1))
        self.log("train_acc", self.train_accuracy, batch_size=self.batch_size)  

        return loss

    def validation_step(self,val_batch,batch_idx):
        
        V, F, VF, FF, Y = val_batch

        Y = Y.squeeze(dim=1)

        x = self((V, F, VF, FF)) 

        #Last loss : for BCE lines
        # x = self.Sigmoid(x).squeeze(dim=1)
        #loss = self.loss(x,Y.to(torch.float32)) 

        loss = self.loss(x,Y) 

        self.log('val_loss', loss, batch_size=self.batch_size)

        predictions = torch.argmax(x, dim=1, keepdim=True)
        self.val_accuracy(predictions.reshape(-1, 1), Y.reshape(-1, 1))
        self.log("val_acc", self.val_accuracy, batch_size=self.batch_size) 


    def test_step(self,test_batch,batch_idx):
        
        V, F, VF, FF, Y = test_batch

        Y = Y.squeeze(dim=1)

        x = self((V, F, VF, FF)) 

        loss = self.loss(x,Y) 

        self.log('test_loss', loss, batch_size=self.batch_size)

        predictions = torch.argmax(x, dim=1, keepdim=True)

        output = [predictions,Y]

        return output

    def test_epoch_end(self,input_test):
        y_pred = []
        y_true = []
        for ele in input_test:
            y_pred += ele[0].tolist()
            y_true += ele[1].tolist()
        target_names = ['V06','V12']

        cf_matrix = confusion_matrix(
            y_true, y_pred
        )

        fig = px.imshow(cf_matrix,labels=dict(x="Predicted condition", y="Actual condition"))
        fig.update_xaxes(side="top")
        fig.write_image("images/confusion_matrix_justIcoConvAvgPooling.png")
        fig.show()

        print(y_pred)
        print(y_true)
        print(classification_report(y_true, y_pred, target_names=target_names))



    def GetView(self,meshes,index):

        phong_renderer = self.phong_renderer.to(self.device)
        R = self.R[index][None].to(self.device)
        T = self.T[index][None].to(self.device)

        _, fragments = phong_renderer(meshes.clone(),R=R,T=T)
        pix_to_face = fragments.pix_to_face   
        pix_to_face = pix_to_face.permute(0,3,1,2)
        return pix_to_face












class BrainIcoAttentionNet(pl.LightningModule):
    def __init__(self,nbr_features,dropout_lvl,image_size,noise_lvl,ico_lvl,batch_size, radius=1.35, lr=1e-4):
        super().__init__()

        self.save_hyperparameters()

        self.nbr_features = nbr_features
        self.dropout_lvl = dropout_lvl
        self.image_size = image_size
        self.noise_lvl = noise_lvl 
        self.batch_size = batch_size
        self.radius = radius

        ico_sphere = utils.CreateIcosahedron(self.radius, ico_lvl)
        ico_sphere_verts, ico_sphere_faces, self.ico_sphere_edges = utils.PolyDataToTensors(ico_sphere)
        self.ico_sphere_verts = ico_sphere_verts
        self.ico_sphere_edges = np.array(self.ico_sphere_edges)
        R=[]
        T=[]
        for coords_cam in self.ico_sphere_verts.tolist():
            camera_position = torch.FloatTensor([coords_cam])
            R_current = look_at_rotation(camera_position)
            # check if camera coords vector and up vector for R are collinear
            #if torch.equal(torch.cross(camera_position,torch.tensor([[0.,1.,0.]])),torch.tensor([[0., 0., 0.]])): 
            #    R = look_at_rotation(camera_position, up = torch.tensor([[0.0, 0.0, 1.0]]),)    
            T_current = -torch.bmm(R_current.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

            R.append(R_current)
            T.append(T_current)
        self.R=torch.cat(R)
        self.T=torch.cat(T)
        self.nbr_cam = len(self.R)

        efficient_net = models.resnet18()
        #efficient_net = models.efficientnet_b0()
        #efficient_net.features[0][0] = nn.Conv2d(self.nbr_features, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        efficient_net.fc = Identity()

        self.drop = nn.Dropout(p=self.dropout_lvl)

        self.noise = GaussianNoise(mean=0.0, std=noise_lvl)
        self.TimeDistributed = TimeDistributed(efficient_net)

        conv2d = nn.Conv2d(512, 256, kernel_size=(3,3),stride=2,padding=0) 
        self.IcosahedronConv2d = IcosahedronConv2d(conv2d,self.ico_sphere_verts,self.ico_sphere_edges)

        self.Attention = SelfAttention(512, 128)
        self.Classification = nn.Linear(256, 2)

        self.Sigmoid = nn.Sigmoid()
        self.loss = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()




        # Initialize a perspective camera.
        self.cameras = FoVPerspectiveCameras()

        
        # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
        raster_settings = RasterizationSettings(
            image_size=self.image_size, 
            blur_radius=0, 
            faces_per_pixel=1, 
            max_faces_per_bin=100000
        )
        # We can add a point light in front of the object. 

        lights = AmbientLights()
        rasterizer = MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings
            )
        self.phong_renderer = MeshRendererWithFragments(
            rasterizer=rasterizer,
            shader=HardPhongShader(cameras=self.cameras, lights=lights)
        )
 
    def forward(self, x):

        V, F, VF, FF = x

        V = V.to(self.device)
        F = F.to(self.device)
        VF = VF.to(self.device)
        FF = FF.to(self.device)

        x, PF = self.render(V,F,VF,FF)

        x = self.noise(x)
        query = self.TimeDistributed(x)
        values = self.IcosahedronConv2d(query)

        x_a, w_a = self.Attention(query, values)
        x_a = self.drop(x_a)
        x = self.Classification(x_a)

        return x

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def render(self,V,F,VF,FF):

        textures = TexturesVertex(verts_features=VF)
        meshes = Meshes(
            verts=V,   
            faces=F, 
            textures=textures
        )

        # To plot the sphere
        # fig = plot_scene({"subplot1":{"meshes":meshes}})
        # fig.show()

        PF = []
        #for i in range(2):  # multiple views of the object
        for i in range(self.nbr_cam): 

            pix_to_face = self.GetView(meshes,i)
            PF.append(pix_to_face.unsqueeze(dim=1))

        PF = torch.cat(PF, dim=1)
        l_features = []
        for index in range(FF.shape[-1]):
            l_features.append(torch.take(FF[:,:,index],PF)*(PF >= 0)) # take each feature
        x = torch.cat(l_features,dim=2)

        return x, PF

    def training_step(self, train_batch, batch_idx):

        V, F, VF, FF, Y = train_batch

        Y = Y.squeeze(dim=1)

        x = self((V, F, VF, FF)) 

        loss = self.loss(x,Y) 

        self.log('train_loss', loss, batch_size=self.batch_size)

        predictions = torch.argmax(x, dim=1, keepdim=True)
        self.train_accuracy(predictions.reshape(-1, 1), Y.reshape(-1, 1))
        self.log("train_acc", self.train_accuracy, batch_size=self.batch_size)  

        return loss

    def validation_step(self,val_batch,batch_idx):
        
        V, F, VF, FF, Y = val_batch

        Y = Y.squeeze(dim=1)

        x = self((V, F, VF, FF)) 

        #Last loss : for BCE lines
        # x = self.Sigmoid(x).squeeze(dim=1)
        #loss = self.loss(x,Y.to(torch.float32)) 

        loss = self.loss(x,Y) 

        self.log('val_loss', loss, batch_size=self.batch_size)

        predictions = torch.argmax(x, dim=1, keepdim=True)
        self.val_accuracy(predictions.reshape(-1, 1), Y.reshape(-1, 1))
        self.log("val_acc", self.val_accuracy, batch_size=self.batch_size) 


    def test_step(self,test_batch,batch_idx):
        
        V, F, VF, FF, Y = test_batch

        Y = Y.squeeze(dim=1)

        x = self((V, F, VF, FF)) 

        loss = self.loss(x,Y) 

        self.log('test_loss', loss, batch_size=self.batch_size)

        predictions = torch.argmax(x, dim=1, keepdim=True)

        output = [predictions,Y]

        return output

    def test_epoch_end(self,input_test):
        y_pred = []
        y_true = []
        for ele in input_test:
            y_pred += ele[0].tolist()
            y_true += ele[1].tolist()
        target_names = ['V06','V12']

        cf_matrix = confusion_matrix(y_true, y_pred)

        fig = px.imshow(cf_matrix,labels=dict(x="Predicted condition", y="Actual condition"),x=target_names,y=target_names)
        fig.update_xaxes(side="top")
        fig.write_image("images/confusion_matrix.png")
        fig.show()

        print(y_pred)
        print(y_true)
        print(classification_report(y_true, y_pred, target_names=target_names))

    def GetView(self,meshes,index):

        phong_renderer = self.phong_renderer.to(self.device)
        R = self.R[index][None].to(self.device)
        T = self.T[index][None].to(self.device)

        _, fragments = phong_renderer(meshes.clone(),R=R,T=T)
        pix_to_face = fragments.pix_to_face   
        pix_to_face = pix_to_face.permute(0,3,1,2)
        return pix_to_face
