
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

|---- README.md       			<- The top-level README for using this project.\
|----data                       <- The directory for storing the data.

	|----raw           			<- for storing original, immutable data dump.\
	|----interim     			<- for storing transformed data.\
	|----processed           	<- for storing canonical dataset for modelling.\
|----figures                    <- The directory for storing the figures and images.\
|----models                     <- The directory for storing saved models.\
|----src                       	<- Python module for this project.

    |----dataset           	    <- Contains code for downloading, preparing, and loading dataset.\
	|----models     			<-Contains pix2pix model architecture.\
	|----experiment           	<- Contains code to perform training.\
    |----testing           		    <- Contains code to perform single image tests.\
|----main.ipynb                 <- Driver notebook for this project.\
|----get_started.py             <- Contains code to make folder structure for this project.\
|----gradio_demo.py             <- Web Interface for testing the model.\
|----requirements.txt           <- Libraries required for this project.



# Img2Img Model - Sobel Filter

To accomplish the task of building a neural network that translates an input image into a Sobel filtered image, Pix2Pix GAN architecture is used. As Pix2Pix model is a conditional generative adversarial network used for image-to-image translation, it felt like this model would not only solve the task but can also be a powerful general-purpose model to derive filter of images. I have also tried standard convolution Autoencoder model but found the model to be overfitting.\
![Example Pix2Pix model](figures/pix2pix.jpeg?raw=true "Pix2Pix Model")
The main dataset used for this project is mini coco dataset from the repo [Mini Coco dataset](https://github.com/giddyyupp/coco-minitrain). Additionally, the code also provides interface to use CIFAR-100 and Oxford IIIT Pets datasets.
The image of the dataset is first processed like grayscale conversion and addition of gaussian blur and because Sobel filter is applied. The input image is the RGB raw image while the output image is grayscale Sobel filtered image. The images are also resized to` 128*128` although the standard pix2pix model uses 256*256, hardware limitations were considered.\
![Sobel Examples](figures/sobek_examples.jpg?raw=true "Sobel Examples")
The standard Pix2Pix architecture was used in this project, except the input channel and output channels were modified to take 3 channels as input and output 1 channel images. With my current hyperparameters the model has converged quickly providing high structural similarity index measure (SSIM) and Peak signal to noise ratio (PSNR). There is room for further optimization.
I have used Pytorch Lightning/Pytorch as my main library. The experiment was done on Cloud GPU platform - Paperspace on P5000 Single GPU.
# Results

For monitoring the experiments W&B (Weights and Bias) tool was used. The code for logging is disabled in this repo as it requires creating account and logging in.\
The training graphs are public and can be found here [W&B Training and Validation Grpahs](https://wandb.ai/xatwik/poly/runs/2eqw2g6t/overview?workspace=user-xatwik).

The focus while monitoring graphs were values of generator and discriminator loss. Training and Validation PSNR and SSIM were the main focus for early stopping.\
![Testing Result](figures/testing.png?raw=true "Testing Result")
The model appears to have converged well and is providing high SSIM and PSNR Values.

# Gradio- Web demo

Web interface for testing the model using `demo_gradio.py`\
Examples for testing can be found in `figures/test1.jpg`
![Gradio Example](figures/gradio.png?raw=true "Gradio Example")


# Img2Img Model - Random Filter (Extra Credits)

Pix2Pix model offers flexible in terms of image-to-image translation. An example of Random Filter being trained is provided below.\
The main limitation I found while training this is hyper parameter optimization which can change with the filters and also judging from the output images, it can be hard can infer the general results of applying a certain filter.

# Answers to set of Questions

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


