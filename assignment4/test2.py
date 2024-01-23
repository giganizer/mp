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
    centerval = array[y-r,x-r]
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
a1t2hesspp =  hessian_points(a1test2, 3, 0.004, 3)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(a1test2, cmap='grey')
plt.subplot(1,2,2)
plt.imshow(a1t2hesspp, cmap="jet")
plt.show()

# %% [markdown]
# its not what its supposed to be, i think - its not like in the instructions
# TODO:
#     continue the instructions and do the next function and the 3 image pairs display
#     if its not like in the instructions take code from git and paste it in and run it and compare
#     identify which part of my code causes the result to differ from the one from git code
#     fix
#     win

# %%
def plot_hess(img, hps):
    """
        hps : hessian points - matrix of determinants that reflects detected points
    """
    plt.imshow(img, cmap="gray")
    xvals = np.nonzero(hps)[1]
    yvals = np.nonzero(hps)[0]
    plt.scatter(xvals, yvals, color="red", s=4, marker="o")
    

# %%
a1visimg = cv2.imread("data\graf\graf_a.jpg", cv2.IMREAD_GRAYSCALE) / 255
a1vthr = 0.0025
a1vbox = 3
a1v1hps = hessian_points(a1visimg, 3, a1vthr, a1vbox)
a1v2hps = hessian_points(a1visimg, 6, a1vthr, a1vbox)
a1v3hps = hessian_points(a1visimg, 9, a1vthr, a1vbox)

plt.figure(figsize=(15,10))
plt.subplot(2,3,1)
plt.title("sigma = 3")
plt.imshow(a1v1hps)
plt.subplot(2,3,2)
plt.title("sigma = 6")
plt.imshow(a1v2hps)
plt.subplot(2,3,3)
plt.title("sigma = 9")
plt.imshow(a1v3hps)
plt.subplot(2,3,4)
plot_hess(a1visimg, a1v1hps)
plt.subplot(2,3,5)
plot_hess(a1visimg, a1v2hps)
plt.subplot(2,3,6)
plot_hess(a1visimg, a1v3hps)
plt.show()

# %% [markdown]
# #### (b)

# %%
def gaussfilter(image, sigma):
    """
        image : input image
        sigma : sigma parameter for generation of Gaussian filter
    """

    #generate gaussian kernel
    gauss_kernel = gauss(sigma)

    #filter along the first dimension
    result = cv2.filter2D(image, -1, gauss_kernel)

    #convolve with transposed kernel
    result = cv2.filter2D(result, -1, gauss_kernel.T)
    
    #return
    return result

# %%
def harris_points(img, sigma, thresh):
    alpha = 0.06
    sigmahat = sigma * 1.6
    box = 3

    ix, iy = part_der(img, sigma)
    c00 = gaussfilter(ix * ix, sigmahat)
    c01 = gaussfilter(ix * iy, sigmahat)
    c10 = np.copy(c01)
    c11 = gaussfilter(iy * iy, sigmahat)
    detc = c00 * c11 - c01 * c10
    tracec = c00 + c11
    points = detc - (alpha * (tracec ** 2))

    pointspp = np.copy(points)
    pointspp[pointspp < thresh] = 0
    for y in range(pointspp.shape[0]):
        for x in range(pointspp.shape[1]):
            if pointspp[y, x] != 0:
                pointspp[y, x] = nms_on_box(points, box, x, y)
    
    return pointspp

# %%
b1visimg = cv2.imread("data\graf\graf_a.jpg", cv2.IMREAD_GRAYSCALE) / 255
b1vthr = 1e-6
b1v1hrp = harris_points(b1visimg, 3, b1vthr)
b1v2hrp = harris_points(b1visimg, 6, b1vthr)
b1v3hrp = harris_points(b1visimg, 9, b1vthr)

plt.figure(figsize=(15,10))
plt.subplot(2,3,1)
plt.title("sigma = 3")
plt.imshow(b1v1hrp)
plt.subplot(2,3,2)
plt.title("sigma = 6")
plt.imshow(b1v2hrp)
plt.subplot(2,3,3)
plt.title("sigma = 9")
plt.imshow(b1v3hrp)
plt.subplot(2,3,4)
plot_hess(b1visimg, b1v1hrp)
plt.subplot(2,3,5)
plot_hess(b1visimg, b1v2hrp)
plt.subplot(2,3,6)
plot_hess(b1visimg, b1v3hrp)
plt.show()


