import cv2
import numpy as np

import skimage.exposure as exposure
from torchvision import transforms


def load_single_image(path, size, device):

    """
    Loading a single image from path along with its ground truth sobel filter
    :param path: path of the single image file
    :param size: for resizing the image
    :param device: "cuda" or "cpu"
    :return: torch input image and its corresponding filtered image
    """

    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)
    ground_truth = apply_sobel_filter(image)

    transform = transforms.Compose([transforms.ToTensor()])

    image = transform(image)
    ground_truth = transform(ground_truth)

    image = image.to(device)
    ground_truth = ground_truth.to(device)

    image = image.unsqueeze(0)
    ground_truth = ground_truth.unsqueeze(0)

    return image, ground_truth


def apply_sobel_filter(image):
    """
    Used by :func load single image

    takes an input image and applies sobel filter
    :ref :https://stackoverflow.com/questions/64743749/sobel-filter-giving-poor-results

    :param image: numpy image
    :return: sobel filtered image
    """

    # convert to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # applying blur
    blur = cv2.GaussianBlur(gray, (0, 0), 1.3, 1.3)

    # apply sobel derivatives on x and y axis
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    # optionally normalize to range 0 to 255 for proper display
    sobelx_norm = exposure.rescale_intensity(sobelx, in_range='image', out_range=(0, 255)).clip(0, 255).astype(np.uint8)
    sobely_norm = exposure.rescale_intensity(sobelx, in_range='image', out_range=(0, 255)).clip(0, 255).astype(np.uint8)

    # square
    sobelx2 = cv2.multiply(sobelx, sobelx)
    sobely2 = cv2.multiply(sobely, sobely)
    # add together and take square root
    sobel_magnitude = cv2.sqrt(sobelx2 + sobely2)
    # normalize to range 0 to 255 and clip negatives
    sobel_magnitude = exposure.rescale_intensity(sobel_magnitude, in_range='image', out_range=(0, 255)).clip(0,
                                                                                                             255).astype(
        np.uint8)

    return sobel_magnitude


