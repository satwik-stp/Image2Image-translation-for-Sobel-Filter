from src.models import pix2pix

from collections import OrderedDict
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision import transforms

import pytorch_lightning as pl


from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim




class Pix2Pix(pl.LightningModule):
    """
    Pytorch Inherited LightningModule for Pix2Pix training and testing
    """
    def __init__(self, hparams,generator,discriminator):
        """
        :param hparams: dictionary of hyperparameters
        :param generator: pix2pix generator
        :param discriminator: pix2pix dicriminator
        """
        super(Pix2Pix, self).__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters()

        self.automatic_optimization = False
        self.generator_lr = self.hparams['generator_lr']  # Generator learning rate
        self.discriminator_lr = self.hparams['discriminator_lr']  # Discriminator learning rate
        self.weight_decay = self.hparams['weight_decay']  # Weight decay e.g. L2 regularization
        self.lr_scheduler_T_0 = self.hparams['lr_scheduler_T_0']  # Optimizer initial restart step number
        self.lr_scheduler_T_mult = self.hparams['lr_scheduler_T_mult']  # Optimizer restart step number factor

        # Models
        self.generator = generator #pix2pix.Generator(dropout_p=self.hparams["generator_dropout_p"])
        self.discriminator = discriminator #pix2pix.Discriminator(dropout_p=self.hparams["discriminator_dropout_p"])

    def forward(self, x):
        return self.generator(x)

    def generator_loss(self, prediction_image, target_image, prediction_label, target_label):
        """
        Pix2pix Generator Loss
        :param prediction_image: predicted image from the generator
        :param target_image: ground truth image
        :param prediction_label: predicted label from the discriminator
        :param target_label: ground truth label
        :return:
        """
        bce_loss = F.binary_cross_entropy(prediction_label, target_label)
        l1_loss = F.l1_loss(prediction_image, target_image)
        mse_loss = F.mse_loss(prediction_image, target_image)
        return bce_loss, l1_loss, mse_loss

    def discriminator_loss(self, prediction_label, target_label):
        """
        Pix2Pix Discriminator Loss
        :param prediction_label: predicted label from the discriminator
        :param target_label: ground truth label
        :return:
        """
        bce_loss = F.binary_cross_entropy(prediction_label, target_label)
        return bce_loss

    def show_images(self, image, output, target, name, n=5):
        """
        :func to show images while the training/validation and testing is being performed
        :param image: input image
        :param output: predicted ground truth image
        :param target: ground truth image
        :param name: name to save the image
        :param n: number of rows
        """

        # selecting only first n images
        image = [i for i in image[:n]]
        output = [i for i in output[:n]]
        target = [i for i in target[:n]]


        grid_top = vutils.make_grid(image, nrow=n)
        grid_mid = vutils.make_grid(target, nrow=n)
        grid_bottom = vutils.make_grid(output, nrow=n)
        grid = torch.cat((grid_top, grid_mid, grid_bottom), 1)

        # Show and Save the first n rows
        plt.imshow(transforms.ToPILImage()(grid.detach().cpu()))
        plt.title("Example image,ground truth and prediction")
        plt.savefig("figures/" + name + ".png")
        plt.show()

    def configure_optimizers(self):
        # Optimizers
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.generator_lr,
                                               weight_decay=self.weight_decay)
        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.discriminator_lr,
                                                   weight_decay=self.weight_decay)
        # Learning Scheduler
        genertator_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(generator_optimizer,
                                                                                       T_0=self.lr_scheduler_T_0,
                                                                                       T_mult=self.lr_scheduler_T_mult)
        discriminator_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(discriminator_optimizer,
                                                                                          T_0=self.lr_scheduler_T_0,
                                                                                          T_mult=self.lr_scheduler_T_mult)
        return [generator_optimizer, discriminator_optimizer], [genertator_lr_scheduler, discriminator_lr_scheduler]

    def training_step(self, batch, batch_idx):

        generator_optimizer, discriminator_optimizer = self.optimizers()
        generator_lr_scheduler, discriminator_lr_scheduler = self.lr_schedulers()

        image, target = batch
        image = image.to(self.hparams["device"])
        target = target.to(self.hparams["device"])

        image_i, image_j = image, image
        target_i, target_j = target, target

        ######################################
        #  Discriminator Loss and Optimizer  #
        ######################################
        # Generator Feed-Forward
        generator_prediction = self.forward(image_i)
        # generator_prediction = torch.clip(generator_prediction, 0, 1)
        # Discriminator Feed-Forward
        discriminator_prediction_real = self.discriminator(torch.cat((image_i, target_i), dim=1))
        discriminator_prediction_fake = self.discriminator(torch.cat((image_i, generator_prediction), dim=1))
        # Discriminator Loss
        discriminator_label_real = self.discriminator_loss(discriminator_prediction_real,
                                                           torch.ones_like(discriminator_prediction_real))
        discriminator_label_fake = self.discriminator_loss(discriminator_prediction_fake,
                                                           torch.zeros_like(discriminator_prediction_fake))
        discriminator_loss = discriminator_label_real + discriminator_label_fake
        # Discriminator Optimizer
        discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        discriminator_optimizer.step()
        discriminator_lr_scheduler.step()

        ##################################
        #  Generator Loss and Optimizer  #
        ##################################
        #  Generator Feed-Forward
        generator_prediction = self.forward(image_j)
        # generator_prediction = torch.clip(generator_prediction, 0, 1)
        # Discriminator Feed-Forward
        discriminator_prediction_fake = self.discriminator(torch.cat((image_j, generator_prediction), dim=1))
        # Generator loss
        generator_bce_loss, generator_l1_loss, generator_mse_loss = self.generator_loss(generator_prediction, target_j,
                                                                                        discriminator_prediction_fake,
                                                                                        torch.ones_like(
                                                                                            discriminator_prediction_fake))
        generator_loss = generator_bce_loss + (generator_l1_loss * self.hparams["lambda"]) + (
                    generator_mse_loss * self.hparams["lambda"])
        # Generator Optimizer
        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()
        generator_lr_scheduler.step()

        # Generator Metrics
        generator_psnr = psnr(generator_prediction, target)
        generator_ssim = ssim(generator_prediction, target)

        # save input and output images at beginning of epoch
        if batch_idx == 0:
            self.show_images(image, generator_prediction, target, "train_input_output")

        # Progressbar and Logging
        loss = OrderedDict({'train_g_bce_loss': generator_bce_loss, 'train_g_l1_loss': generator_l1_loss,
                            'train_g_mse_loss': generator_mse_loss,
                            'train_g_loss': generator_loss, 'train_d_loss': discriminator_loss,
                            'train_g_psnr': generator_psnr, 'train_g_ssim': generator_ssim,
                            'train_g_lr': generator_lr_scheduler.get_last_lr()[0],
                            'train_d_lr': discriminator_lr_scheduler.get_last_lr()[0]})
        self.log_dict(loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, target = batch
        image = image.to(self.hparams["device"])
        target = target.to(self.hparams["device"])

        # Generator Feed-Forward
        generator_prediction = self.forward(image)
        # generator_prediction = torch.clip(generator_prediction, 0, 1)
        # Generator Metrics
        generator_psnr = psnr(generator_prediction, target)
        generator_ssim = ssim(generator_prediction, target)

        # Progressbar and Logging
        metrics = OrderedDict({'val_g_psnr': generator_psnr, 'val_g_ssim': generator_ssim})

        self.log_dict(metrics, prog_bar=True)
        return metrics

    def validation_epoch_end(self, outputs):
        val_g_psnr = torch.stack([x['val_g_psnr'] for x in outputs]).mean()
        val_g_ssim = torch.stack([x['val_g_ssim'] for x in outputs]).mean()
        
    
        logs = {"avg_val_g_psnr": val_g_psnr, "avg_val_g_ssim": val_g_ssim}
        self.log_dict(logs, prog_bar=True)
    
        return logs

    def test_step(self, batch, batch_idx):
        image, target = batch
        image = image.to(self.hparams["device"])
        target = target.to(self.hparams["device"])

        # Generator Feed-Forward
        generator_prediction = self.forward(image)
        # generator_prediction = torch.clip(generator_prediction, 0, 1)
        # Generator Metrics
        generator_psnr = psnr(generator_prediction, target)
        generator_ssim = ssim(generator_prediction, target)

        # save input and output images at beginning of epoch
        if batch_idx == 0:
            self.show_images(image, generator_prediction, target, "test_input_output")

        # Progressbar and Logging
        metrics = OrderedDict({'test_g_psnr': generator_psnr, 'test_g_ssim': generator_ssim})

        self.log_dict(metrics, prog_bar=True)
        return metrics

    def test_epoch_end(self, outputs):
        test_g_psnr = torch.stack([x['test_g_psnr'] for x in outputs]).mean()
        test_g_ssim = torch.stack([x['test_g_ssim'] for x in outputs]).mean()
    
        logs = {"avg_test_g_psnr": test_g_psnr, "avg_test_g_ssim": test_g_ssim}
        self.log_dict(logs, prog_bar=True)
    
        return logs
