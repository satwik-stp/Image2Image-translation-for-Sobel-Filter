
# Getting Started 

To get started with this project:

Installing the libraries:\
`pip install -r requirements.txt`

Creating the folder structure:\
`python setup.py`

Download trained model from:\
`https://drive.google.com/file/d/1ja1ZPJsW-KW6_Jm7p9L5tkvmbx2t3aiN/view?usp=sharing` \
and place it in `models` subdirectory

The main driver of simple autoencoder notebook can be found:\
`simple_main.ipynb`

The main driver of Pix2Pix notebook can be found:\
`main.ipynb`

For a quick demo of the model in a web interface:\
`python demo_gradio.py`


# Project structure

|---- README.md       		<- The top-level README for using this project.\
|----data                       <- The directory for storing the data.

	|----raw           	<- for storing original, immutable data dump.\
	|----interim     	<- for storing transformed data.\
	|----processed          <- for storing canonical dataset for modelling.\
|----figures                    <- The directory for storing the figures and images.\
|----models                     <- The directory for storing saved models.\
|----src                       	<- Python module for this project.

    |----dataset           	<- Contains code for downloading, preparing, and loading dataset.\
	|----models     	<-Contains pix2pix model architecture.\
	|----experiment         <- Contains code to perform training.\
    |----testing           	<- Contains code to perform single image tests.\
|----main.ipynb                 <- Driver notebook for Pix2Pix.\
|----simple_main.ipynb          <- Driver notebook for Simple Autoencoder.\
|----get_started.py             <- Contains code to make folder structure for this project.\
|----gradio_demo.py             <- Web Interface for testing the model.\
|----requirements.txt           <- Libraries required for this project.



# Img2Img Model - Sobel Filter

This project contains code for simple model based on single layer of convolution and single layer of deconvolution, and also contains code for an advanced architecture
-Pix2Pix. 

## Simple Convolution Autoencoder

As I understand, at the core of this task is to find an approximation to Sobel Filter using Deep Learning. From this it can be inferred that the network doesnt require, lots of layers to train.

First part of this project is 2 layers of convolutions followed by 2 layers of deconvolution (to maintain the shape of the output image).

```
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.batch_norm1=nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.batch_norm2=nn.BatchNorm2d(32)
        
        self.trans_conv1=torch.nn.ConvTranspose2d(32,16,3)
        self.batch_norm3=nn.BatchNorm2d(16)
        
        self.trans_conv2=torch.nn.ConvTranspose2d(16,1,3)
        self.batch_norm4=nn.BatchNorm2d(1)


    def forward(self, x):
        x = self.conv1(x)
        x= F.relu(self.batch_norm1(x))
        
        x = self.conv2(x)
        x= F.relu(self.batch_norm2(x))
        
        x = self.trans_conv1(x)
        x= F.relu(self.batch_norm3(x))
        
        x = self.trans_conv2(x)
        x= F.relu(self.batch_norm4(x))
        return x
```

## Pix2Pix

To accomplish the task of building a neural network that translates an input image into a Sobel filtered image using an advanced archiecture, Pix2Pix GAN architecture is used. As Pix2Pix model is a conditional generative adversarial network used for image-to-image translation, it felt like this model would not only solve the task but can also be a powerful general-purpose model to derive filter of images. I have also tried standard convolution Autoencoder model but found the model to be overfitting.\
![Example Pix2Pix model](figures/pix2pix.jpeg?raw=true "Pix2Pix Model")

The standard Pix2Pix architecture was used in this project, except the input channel and output channels were modified to take 3 channels as input and output 1 channel images. With my current hyperparameters the model has converged quickly providing high structural similarity index measure (SSIM) and Peak signal to noise ratio (PSNR). There is room for further optimization.
I have used Pytorch Lightning/Pytorch as my main library. The experiment was done on Cloud GPU platform - Paperspace on P5000 Single GPU.

## Dataset
The main dataset used for this project is mini coco dataset from the repo [Mini Coco dataset](https://github.com/giddyyupp/coco-minitrain). Additionally, the code also provides interface to use CIFAR-100 and Oxford IIIT Pets datasets.
The image of the dataset is first processed like grayscale conversion and addition of gaussian blur and because Sobel filter is applied. The input image is the RGB raw image while the output image is grayscale Sobel filtered image. The images are also resized to `64*64` for Autoencoder and ` 128*128` for Pix2Pix model although the standard pix2pix model uses 256*256, hardware limitations were considered.\
![Sobel Examples](figures/sobek_examples.jpg?raw=true "Sobel Examples")


# Results

## Autoencoder

Testing result for the Autoencoder is given below, it has produced decent ssim, psnr values\
```
Test Loss: 0.05406247079372406
Test PSNR: 21.546340942382812
Test SSIM: 0.8205837607383728
```

![Testing Result1](figures/autoencoder_graph.png?raw=true "Testing Result 1")\
![Testing Result2](figures/autoencoder_graph1.png?raw=true "Testing Result 2")\
![Testing Result3](figures/autoencoder_graph2.png?raw=true "Testing Result 3")


## Pix2Pix
For monitoring the experiments W&B (Weights and Bias) tool was used. The code for logging is disabled in this repo as it requires creating account and logging in.\
The training graphs are public and can be found here [W&B Training and Validation Grpahs](https://wandb.ai/xatwik/poly/runs/2eqw2g6t/overview?workspace=user-xatwik).

The focus while monitoring graphs were values of generator and discriminator loss. Training and Validation PSNR and SSIM were the main focus for early stopping.\
![Testing Result](figures/testing.png?raw=true "Testing Result")
The model appears to have converged well and is providing high SSIM and PSNR Values.

### Gradio- Web demo

Web interface for testing the model using `demo_gradio.py`\
Examples for testing can be found in `figures/test1.jpg`
![Gradio Example](figures/gradio.png?raw=true "Gradio Example")


# Img2Img Model - Random Filter (Extra Credits)

Autoencoder model to be described here soon

Pix2Pix model offers flexible in terms of image-to-image translation. An example of Random Filter being trained is provided below.\
The main limitation I found while training this is hyper parameter optimization which can change with the filters and also judging from the output images, it can be hard can infer the general results of applying a certain filter.

# Answers to set of Questions

## Autoencoder
**1.What if the image is really large or not of a standard size?**\
The images are resized to 64*64 dimensions (128*128 works too). If images are really large, random cropping can also be used.

**2.What should occur at the edges of the image?**\
The Sobel filter is used for edge detection. It works by calculating the gradient of image intensity at each pixel within the image. 
As for the frame of image itself, unless there is observable border, I have not noticed a edge of the image frame forming.

**3.Are you using a fully convolutional architecture?**\
I am using autoencoder architecture whose layers include convolutional, convolutional transpose layers.

**4.Are there optimizations built into your framework of choice (e.g. Pytorch) that can make this fast?**\
For making the training faster, network size, learning rate or batch size can be increased. Additionally Pytorch supports Multi GPU training but I have not used because of lack of hardware.

**5.What if you wanted to optimize specifically for model size?**\
I have obsvered from experimentations that 2 layer convolution works better than single layer. Therefore isnt much of a leap with 3 layered network except for faster learning capabilites. Therefore with the view of hardware limitation, 2 layered convolution network was the most optimal for me.

**6.How do you know when training is complete?**\
From combination of training and validation MSE loss, SSIM, PSNR graphs. This autoencoder model is observed to converge well. 

**7.What is the benefit of a deeper model in this case? When might there be a benefit for a deeper model (not with a sobel kernel but generally when thinking about image to image transformations)?**\
I have obsvered from experimentations that 2 layer convolution works better than single layer. Therefore isnt much of a leap with 3 layered network except for faster learning capabilites. Therefore, there is improvement with increasing the layers but diminishing outcomes after a certain point.



## Pix2Pix
**1.What if the image is really large or not of a standard size?**\
The images are resized to 128*128 dimensions (256*256 works too). Incase images are really large, cropping can also be performed.

**2.What should occur at the edges of the image?**\
The Sobel filter is used for edge detection. It works by calculating the gradient of image intensity at each pixel within the image. 
As for the frame of image itself, unless there is observable border, I have not noticed a edge of the image frame forming.

**3.Are you using a fully convolutional architecture?**\
I am using PixPix2 model whose building blocks are convolution layers.

**4.Are there optimizations built into your framework of choice (e.g. Pytorch) that can make this fast?**\
Optimization I used for this project include dropout, learning scheduler. For making the training faster, learning rate or batch size can be increased. Additionally Pytorch supports Multi GPU training but I have not used because of lack of hardware.

**5.What if you wanted to optimize specifically for model size?**\
For larger models, large batch size can be limiting therefore, I have reduced the batch size. The model might perform well with lesser epochs and lower learning rates.

**6.How do you know when training is complete?**\
Since GAN training is very unstable, I have relied on training and validation PSNR and SSIM values. Training is stopped when these values have stablized and converged. 

**7.What is the benefit of a deeper model in this case? When might there be a benefit for a deeper model (not with a sobel kernel but generally when thinking about image to image transformations)?**\
I have used convolutional Autoencoder as my first model, which didnt not produce high SSIM and PSNR values and also overfit on data. Pix2Pix is a deeper model, I have not noticed overfitting, it has produced high PSNR and SSIM values and converged quickly.


