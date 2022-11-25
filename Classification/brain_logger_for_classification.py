from pytorch_lightning.callbacks import Callback
import torchvision
import torch


class BrainNetImageLogger(Callback):
    def __init__(self,num_features = 3 , num_images=12, log_steps=10,mean=0,std=0.015):
        self.num_features = num_features
        self.log_steps = log_steps
        self.num_images = num_images
        self.mean = mean
        self.std = std

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):        

        if batch_idx % self.log_steps == 0:

                V, F, VF, FF, Y = batch

                V = V.to(pl_module.device, non_blocking=True)
                F = F.to(pl_module.device, non_blocking=True)
                VF = VF.to(pl_module.device, non_blocking=True)
                FF = FF.to(pl_module.device, non_blocking=True)
                Y = Y.to(pl_module.device, non_blocking=True)

                with torch.no_grad():

                    images, PF = pl_module.render(V, F, VF, FF)

                    grid_images = torchvision.utils.make_grid(images[0, 0:self.num_images, 0:self.num_features, :, :])
                    trainer.logger.experiment.add_image('Image features', grid_images, pl_module.global_step)

                    images_noiseM = images +  torch.normal(self.mean, self.std,size=images.shape, device=images.device)*(images!=0)

                    grid_images_noiseM = torchvision.utils.make_grid(images_noiseM[0, 0:self.num_images, 0:self.num_features, :, :])
                    trainer.logger.experiment.add_image('Image + noise M ', grid_images_noiseM, pl_module.global_step)


                    # grid_eacsf = torchvision.utils.make_grid(images[0, 0:self.num_images, 0:1, :, :])
                    # trainer.logger.experiment.add_image('Image eacsf', grid_eacsf, pl_module.global_step)

                    # grid_sa = torchvision.utils.make_grid(images[0, 0:self.num_images, 1:2, :, :])
                    # trainer.logger.experiment.add_image('Image sa', grid_sa, pl_module.global_step)

                    # grid_thickness = torchvision.utils.make_grid(images[0, 0:self.num_images, 1:2, :, :])
                    # trainer.logger.experiment.add_image('Image thickness', grid_thickness, pl_module.global_step)
