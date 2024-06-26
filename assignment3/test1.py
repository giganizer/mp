# %%
#imports
import numpy as np
from a3_utils import *
import cv2
from matplotlib import pyplot as plt
import math

# %% [markdown]
# ### Exercise 1: Image derivates

# %% [markdown]
# #### (a)
# 
# -> OneNote

# %% [markdown]
# #### (b)

# %%
def gaussdx(sigma):
    size = 2 * math.ceil((3 * sigma)) + 1

    vals = np.arange(-(size//2),size//2+1)

    kernel = -(
        (
            (1)
            /
            (math.sqrt(2*math.pi) * sigma ** 3)
        )
        *
        vals
        *
        np.exp(
            -(
                (vals**2)
                /
                (2 * sigma **2)
            )
        )
    )

    kernel = kernel / np.sum(np.absolute(kernel))

    return kernel

# %% [markdown]
# #### (c)

# %%
def gauss(sigma):
    size = 2 * math.ceil((3 * sigma)) + 1
    vals = np.arange(-(size//2),size//2+1)
    kernel = (1 / math.sqrt(2 * math.pi) * sigma) * np.exp(-((vals**2) / (2 * sigma**2)))
    kernel = kernel / np.sum(kernel)
    return kernel

# %%
impulse = np.zeros((50, 50))
impulse[25, 25] = 1

e1c_sigma = 4
e1c_g = gauss(e1c_sigma)
e1c_g = e1c_g.reshape(1, -1)
e1c_d = gaussdx(e1c_sigma)
e1c_d = e1c_d.reshape(1, -1)
e1c_g = np.flip(e1c_g)
e1c_d = np.flip(e1c_d)

e1c_result_g_gt = cv2.filter2D(impulse, -1, e1c_g)
e1c_result_g_gt = cv2.filter2D(e1c_result_g_gt, -1, e1c_g.T)

e1c_result_g_dt = cv2.filter2D(impulse, -1, e1c_g)
e1c_result_g_dt = cv2.filter2D(e1c_result_g_dt, -1, e1c_d.T)

e1c_result_d_gt = cv2.filter2D(impulse, -1, e1c_d)
e1c_result_d_gt = cv2.filter2D(e1c_result_d_gt, -1, e1c_g.T)

e1c_result_gt_d = cv2.filter2D(impulse, -1, e1c_g.T)
e1c_result_gt_d = cv2.filter2D(e1c_result_gt_d, -1, e1c_d)

e1c_result_dt_g = cv2.filter2D(impulse, -1, e1c_d.T)
e1c_result_dt_g = cv2.filter2D(e1c_result_dt_g, -1, e1c_g)


plt.figure(figsize=(16,8))
plt.subplot(2,3,1)
plt.title("Impulse")
plt.imshow(impulse, cmap="gray")
plt.subplot(2,3,2)
plt.title("b) G, D^T")
plt.imshow(e1c_result_g_dt, cmap="gray")
plt.subplot(2,3,3)
plt.title("c) D, G^T")
plt.imshow(e1c_result_d_gt, cmap="gray")
plt.subplot(2,3,4)
plt.title("a) G, G^T")
plt.imshow(e1c_result_g_gt, cmap="gray")
plt.subplot(2,3,5)
plt.title("d) G^T, D")
plt.imshow(e1c_result_gt_d, cmap="gray")
plt.subplot(2,3,6)
plt.title("e) D^T, G")
plt.imshow(e1c_result_dt_g, cmap="gray")
plt.show()

# %% [markdown]
# The order of operations is not important, as can be seen in the images.

# %% [markdown]
# #### (d)

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
def gradient_magnitudes(I, sigma):
    Ix, Iy = part_der(I, sigma)
    m = np.sqrt(np.square(Ix) + np.square(Iy))
    phi = np.arctan2(Iy, Ix)
    return m, phi

# %%
e1d_museum = cv2.imread("images\museum.jpg", cv2.IMREAD_GRAYSCALE)
e1d_museum = e1d_museum.astype(np.float64)
# e1d_museum = e1d_museum / 255

e1d_sigma = 2
e1d_ix, e1d_iy = part_der(e1d_museum, e1d_sigma)
e1d_ixx, e1d_iyy, e1d_ixy = part_2nd_ord_der(e1d_museum, e1d_sigma)
e1d_imag, e1d_idir = gradient_magnitudes(e1d_museum, e1d_sigma)

plt.figure(figsize=(16,6))
plt.subplot(2,4,1)
plt.title("Original")
plt.imshow(e1d_museum, cmap="gray")
plt.subplot(2,4,2)
plt.title("I_x")
plt.imshow(e1d_ix, cmap="gray")
plt.subplot(2,4,3)
plt.title("I_y")
plt.imshow(e1d_iy, cmap="gray")
plt.subplot(2,4,4)
plt.title("I_mag")
plt.imshow(e1d_imag, cmap="gray")
plt.subplot(2,4,5)
plt.title('I_xx')
plt.imshow(e1d_ixx, cmap='gray')
plt.subplot(2,4,6)
plt.title('I_xy')
plt.imshow(e1d_ixy, cmap='gray')
plt.subplot(2,4,7)
plt.title('I_yy')
plt.imshow(e1d_iyy, cmap='gray')
plt.subplot(2,4,8)
plt.title('I_dir')
plt.imshow(e1d_idir, cmap='gray')
plt.show()

# %% [markdown]
# ### Exercise 2: Edges in images

# %% [markdown]
# #### (a)

# %%
def findedges(I, sigma, theta):
    Imag, _ = gradient_magnitudes(I, sigma)
    Ie = np.where(Imag >= theta, 1, 0)
    return Ie

# %%
e2a_museum = cv2.imread("images\museum.jpg", cv2.IMREAD_GRAYSCALE) / 255

plt.figure(figsize=(16,6))
plt.subplot(2,4,1)
plt.title("Original")
plt.imshow(e2a_museum, cmap="gray")
for i in range(1,8):
    e2a_step = 2.5
    e2a_offset = 10
    plt.subplot(2,4,i+1)
    plt.title('{}'.format(i*e2a_step+e2a_offset))
    plt.imshow(findedges(e2a_museum, 2, (i*e2a_step+e2a_offset)/180), cmap='gray')
plt.show()

# %% [markdown]
# #### (b)

# %%
def non_maxima(mags, angls, thr):
    ret = np.copy(mags)
    neighborhood_size = 1

    for y in range(mags.shape[0]):
        for x in range(mags.shape[1]):
            x_tl = x - neighborhood_size
            x_tl = 0 if x_tl < 0 else x_tl
            y_tl = y - neighborhood_size
            y_tl = 0 if y_tl < 0 else y_tl
            x_br = x + neighborhood_size
            x_br = mags.shape[1]-1 if x_br > mags.shape[1]-1 else x_br
            y_br = y + neighborhood_size
            y_br = mags.shape[0]-1 if y_br > mags.shape[0]-1 else y_br

            current = mags[y,x]
            cut_mags = np.copy(mags[y_tl:y_br, x_tl:x_br])
            cut_angls = np.copy(angls[y_tl:y_br, x_tl:x_br])
            
            cut_angls = np.abs(cut_angls - current)/current
            cut_mags = np.where(cut_angls <= thr, 0, cut_mags)

            largest = np.max(cut_mags)

            if current != largest: ret[y,x] = 0

    return ret

# %%
e2b_museum = cv2.imread("images\museum.jpg", cv2.IMREAD_GRAYSCALE)
e2b_museum = e2b_museum.astype(np.float64)

e2b_thr = 0.16
e2b_test1_mag, e2b_test1_dir = gradient_magnitudes(e2b_museum, 2)

#?
e2b_test1_mag = (e2b_test1_mag - np.min(e2b_test1_mag)) / (np.max(e2b_test1_mag) - np.min(e2b_test1_mag))
e2b_test1_mag = np.where(e2b_test1_mag < e2b_thr, 0, e2b_test1_mag)

e2b_test1_nms = non_maxima(e2b_test1_mag, e2b_test1_dir, e2b_thr)

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(e2b_museum, cmap="gray")
plt.subplot(1,3,2)
plt.imshow(e2b_test1_mag, cmap="gray")
plt.subplot(1,3,3)
plt.imshow(e2b_test1_nms, cmap="gray")


# %% [markdown]
# ### Exercise 3: Detecting lines

# %% [markdown]
# Question --> OneNote

# %% [markdown]
# #### (a)

# %%
e3a_accumulators = []
e3a_points = [
    (10, 10),
    (30, 60),
    (50, 20),
    (80, 90)
]

for i in range(len(e3a_points)):
    point = e3a_points[i]
    e3a_accumulators.append(np.zeros((300,300)))
    for theta in range(e3a_accumulators[i].shape[1]):
        theta_rad = ((theta-150)/150)*math.pi
        ro = round(
            point[0] # x
            * math.cos(theta_rad) 
            + point[1] # y
            * math.sin(theta_rad)
            )
        # print(theta_rad, " | ", ro)
        e3a_accumulators[i][ro-150][theta] += 1
    
plt.figure(figsize=(16,16))
for i in range(len(e3a_points)):
    plt.subplot(2,2,i+1)
    plt.title("x = {}, y = {}".format(e3a_points[i][0], e3a_points[i][1]))
    plt.imshow(e3a_accumulators[i])
plt.show()

# %% [markdown]
# #### (b)

# %%
def hough_find_lines(img, bins, th):
    """
    img : binary image
    bins : tuple (<number of theta bins>, <number of rho bins>)
    th : threshold
    """
    theta_bins, rho_bins = bins

    theta = np.linspace(-np.pi/2, np.pi/2, theta_bins)
    D = math.sqrt(
        img.shape[0]**2
        +
        img.shape[1]**2
    )
    rho = np.linspace(-D, D, rho_bins)

    A = np.zeros((rho_bins, theta_bins))

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y][x] != 0:
                #nonzero pixel
                for i_th in range(len(theta)):
                    rho_val = (
                          x # x
                        * math.cos(theta[i_th]) 
                        + y # y
                        * math.sin(theta[i_th])
                    )
                    rho_val_bin = np.digitize(rho_val, rho)-1
                    A[rho_val_bin][i_th] += 1
    
    return A

# %%
e3b_testimg_100 = np.zeros((100,100))
e3b_testimg_100[9,9], e3b_testimg_100[9,19] = 1, 1
e3b_hough_1 = hough_find_lines(e3b_testimg_100, (180, 180), None)
plt.imshow(e3b_hough_1)


