# %%
#imports
import numpy as np
from a4_utils import *
import cv2
from matplotlib import pyplot as plt
import math
import os

# %% [markdown]
# ### Exercise 1: Feature points detectors

# %% [markdown]
# #### (a)

# %%
def gauss(sigma):
    size = 2 * math.ceil((3 * sigma)) + 1
    vals = np.arange(-(size//2),size//2+1)
    kernel = (1 / math.sqrt(2 * math.pi) * sigma) * np.exp(-((vals**2) / (2 * sigma**2)))
    kernel = kernel / np.sum(kernel)
    return kernel

# %%
def part_der(img, sigma, dev=False):
    g = np.flip(gauss(sigma).reshape(1, -1))
    dg = np.flip(gaussdx(sigma).reshape(1, -1))

    # "however, we have to remember to always filter the image before we perform derivation"  - I remembered
    Ix = cv2.filter2D(img, -1, g.T)
    Ix = cv2.filter2D(img, -1, dg)
    Iy = cv2.filter2D(img, -1, g)
    Iy = cv2.filter2D(img, -1, dg.T)

    if dev:
        return Ix, Iy, g, dg # dev mode : return kernels also
    else:
        return Ix, Iy

# %%
def part_2nd_ord_der(img, sigma):
    Ix, Iy, g, dg = part_der(img, sigma, dev=True)
    
    Ixx = cv2.filter2D(Ix, -1, g.T)
    Ixx = cv2.filter2D(Ixx, -1, dg)

    Iyy = cv2.filter2D(Iy, -1, g)
    Iyy = cv2.filter2D(Iyy, -1, dg.T)

    Ixy = cv2.filter2D(Ix, -1, g)
    Ixy = cv2.filter2D(Ixy, -1, dg.T)

    return Ixx, Iyy, Ixy

# %%
def hessian_points(img, sigma):
    ixx, ixy, iyy = part_2nd_ord_der(img,sigma)
    dets = ixx * iyy - (ixy ** 2) #determinants
    return dets

# %%
#"Test the function using image from graf_a.jpg as your input (do not forget to convert it to grayscale) and visualize the result."
a1test1 = cv2.imread("data\graf\graf_a.jpg", cv2.IMREAD_GRAYSCALE) / 255
a1t1hess =  hessian_points(a1test1, 3)

plt.subplot(1,2,1)
plt.imshow(a1test1, cmap='grey')
plt.subplot(1,2,2)
plt.imshow(a1t1hess)
plt.show()

# %%
def nms_on_box(array, neighborhood, x, y):
    """
        Performs non-maximum suppression for the center pixel in a box, considering all the pixels in the cutout (neighborhood).
        Returns: center pixel value after suppression (so either its original value or 0).
        Params:
            array : array to perform nms on
            neighborhood : size of neighborhood - total length, must be odd
            x : x coordinate of center pixel
            y : y coordinate of center pixel
    """
    r = int(neighborhood/2)
    x = x + r
    y = y + r
    padded = np.pad(array, r, mode="constant", constant_values=0)
    box_cutout = padded[y-r:y+r+1, x-r:x+r+1]
    centerval = array[y,x]
    return (0 if np.max(box_cutout) > centerval else centerval)

# %%
def hessian_points(img, sigma, thresh, box):
    """
        params:
            img : image
            sigma : sigma for gaussian filter in derivative
            thresh : threshold for post processing step
            box : neighborhood size for post processing step (length of a square's side)
    """
    ixx, ixy, iyy = part_2nd_ord_der(img,sigma)
    dets = ixx * iyy - (ixy ** 2) #determinants
    detsnms = np.copy(dets)
    detsnms[detsnms < thresh] = 0 #threshold first
    for y in range(dets.shape[0]):
        for x in range(dets.shape[1]):
            if detsnms[y, x] != 0:
                detsnms[y, x] = nms_on_box(dets, box, x, y) #only do nms for values that arent already 0

    return detsnms

# %%
a1test2 = cv2.imread("data\graf\graf_a.jpg", cv2.IMREAD_GRAYSCALE) / 255
a1t2hesspp =  hessian_points(a1test2, 3, 0.004, 9)

plt.subplot(1,2,1)
plt.imshow(a1test2, cmap='grey')
plt.subplot(1,2,2)
plt.imshow(a1t2hesspp)
plt.show()


