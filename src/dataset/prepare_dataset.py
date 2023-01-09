import os
import random

import cv2
import numpy as np

import skimage.exposure as exposure
from sklearn.model_selection import train_test_split\

from scipy import ndimage

"""
This submodule provides code for preparing the downloaded datasets into input and output filtered image  
"""

def prepare_img_and_gt_folder(folder,img_size,filter,save):

    """
    :func takes folder containing input image files and extracts the sobel/random filter
    returns the corresponding input and sobel/random

    :param folder: folder containing input image files
    :param img_size: resizing the image
    :param filter: type of filter
                    can be "sobel" or "random"
    :param save: boolean for save the transformed data in data->interim
    :return: input image and its corresponding ground truth
    """

    images=[]
    ground_truths=[]

    for filename in os.listdir(folder):
        
        try:

            # load the image using opencv
            image=load_image(folder=folder,filename=filename,size=[img_size,img_size])

            # apply filter to obtain the ground_truth
            if filter=="sobel":
                ground_truth=apply_sobel_filter(image)
            elif filter=="random":
                ground_truth = apply_random_filter(image)

            if (image is not None) and (ground_truth is not None):
                images.append(image)
                ground_truths.append(ground_truth)
        except:
            continue

    # using numpy array instead torch tensor as it takes lesser space
    images=np.array(images)
    ground_truths=np.array(ground_truths)

    if save:
        np.save('data/interim/images',images)
        np.save('data/interim/ground_truths',ground_truths)

    return images,ground_truths


def load_image(folder,filename,size):

    """
    Used by :func prepare_img_and_gt_folder
    loads the image , coverts from bgr to rgb and resizes the image

    :param folder: root directory of the image
    :param filename: image file name
    :param size: image resizing parameters
    :return: loaded and resized image
    """

    image = cv2.imread(os.path.join(folder,filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)

    return image

def split_data(images,gts,test_size,val_size):

    """
    :func to split the images and ground_truth datasets for train, testing and validation

    :param images: numpy/list/torch of input images
    :param gts: numpy/list/torch of ground truth images
    :param test_size: size of test split from entire dataset
    :param val_size: size of val split from experiment dataset
    :return: images and ground truths split into images,
    """


    X_train, X_test, y_train, y_test = train_test_split(images, gts, test_size=test_size, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=1)
    
    return X_train,X_val,X_test,y_train,y_val,y_test


def randomly_select(images, gts,n):
    """
    :func to randomly select portion of dataset with the view of GPU memory limitation

    :param images: numpy/list/torch of input images
    :param gts: numpy/list/torch of ground truth images
    :param n: size of the portion
    :return: randomly selected n length images and corresponding ground truths images
    """

    comb = list(zip(images, gts))
    random.shuffle(comb)

    images, gts = zip(*comb)

    images = images[:n]
    gts = gts[:n]

    return images,gts


def apply_sobel_filter(image):

    """
    Used by :func  prepare_img_and_gt_folder

    takes an input image and applies sobel filter
    :ref :https://stackoverflow.com/questions/64743749/sobel-filter-giving-poor-results

    :param image: numpy image
    :return: sobel filtered image
    """


    # convert to gray
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # applying blur
    blur = cv2.GaussianBlur(gray, (0,0), 1.3, 1.3)

    # apply sobel derivatives on x and y axis
    sobelx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=3)
    # optionally normalize to range 0 to 255 for proper display
    sobelx_norm= exposure.rescale_intensity(sobelx, in_range='image', out_range=(0,255)).clip(0,255).astype(np.uint8)
    sobely_norm= exposure.rescale_intensity(sobelx, in_range='image', out_range=(0,255)).clip(0,255).astype(np.uint8)

    # square 
    sobelx2 = cv2.multiply(sobelx,sobelx)
    sobely2 = cv2.multiply(sobely,sobely)
    # add together and take square root
    sobel_magnitude = cv2.sqrt(sobelx2 + sobely2)
    # normalize to range 0 to 255 and clip negatives
    sobel_magnitude = exposure.rescale_intensity(sobel_magnitude, in_range='image', out_range=(0,255)).clip(0,255).astype(np.uint8)

    return sobel_magnitude


def apply_random_filter(image):

    """
    Used by :func prepare_img_and_gt_folder

    takes an input image and applies a random filter
    :ref :http://www.adeveloperdiary.com/data-science/computer-vision/how-to-implement-sobel-edge-detection-using-python-from-scratch/

    :param image: numpy image
    :return: random filtered image
    """


    # convert to gray
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # applying blur
    image = cv2.GaussianBlur(gray, (0,0), 1.3, 1.3)
    
    kernel=np.random.rand(3,3)
    output=ndimage.convolve(image,kernel,mode="constant")

    return output




