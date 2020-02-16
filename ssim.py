# Usage:
#
# python3 script.py --input original.png --output modified.png
# Based on: https://github.com/mostafaGwely/Structural-Similarity-Index-SSIM-

# 1. Import the necessary packages
from skimage.measure import compare_ssim
import argparse
#import imutils
#import cv2
import numpy
from skimage import io
import tensorflow as tf
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-image1')
parser.add_argument('-image2')

args = parser.parse_args()
def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def psnr(im1, im2):
    img_arr1 = numpy.array(im1).astype('float32')
    img_arr2 = numpy.array(im2).astype('float32')
    mse = tf.reduce_mean(tf.squared_difference(img_arr1, img_arr2))
    psnr = tf.constant(255**2, dtype=tf.float32)/mse
    result = tf.constant(10, dtype=tf.float32)*log10(psnr)
    with tf.Session():
        result = result.eval()
    return result
def mse(img1,img2):
    mserr=np.sum((img1.astype("float")-img2.astype("float"))**2)
    mserr /=float(img1.shape[0]*img1.shape[1])
    return mserr
def load_image(infilename) :
    img = Image.open(infilename)
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

# 2. Construct the argument parse and parse the arguments
"""ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True, help="Directory of the image that will be compared")
ap.add_argument("-s", "--second", required=True, help="Directory of the image that will be used to compare")
args = vars(ap.parse_args())"""

# 3. Load the two input images
image1=load_image(args.image1)
image2=load_image(args.image2)
img_arr1 = numpy.array(image1).astype('float32')
img_arr2 = numpy.array(image2).astype('float32')
# 4. Convert the images to grayscale
#grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
#grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
# 5. Compute the Structural Similarity Index (SSIM) between the two
#    images, ensuring that the difference image is returned
(score, diff) = compare_ssim(np.dot(img_arr1[...,:3], [0.2989, 0.5870, 0.1140]), np.dot(img_arr2[...,:3], [0.2989, 0.5870, 0.1140]), full=True)
diff = (diff * 255).astype("uint8")
print("PSNR: {}".format(psnr(image1,image2)))
# 6. You can print only the score if you want
print("SSIM: {}".format(score))
print("MSE: {}".format(mse(img_arr1,img_arr2)))
