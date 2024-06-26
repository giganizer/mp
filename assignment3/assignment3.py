# %%
#imports
import numpy as np
from a3_utils import *
import cv2
from matplotlib import pyplot as plt
import math
import os

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
# #### (e)

# %%
def img_retrieval(dir, bins):
    """
        dir : directory name
        bins : number of bins for histogram
    """
    #declare main data struct
    objects_data = {}

    #iterate over file names in dir
    for filename in os.listdir(dir):
        #init object data struct
        object_instance = {
            #"filename" : filename, #self-explanatory
            "image" : None, #image matrix
            "feature" : None
        }
        #read image
        object_instance["image"] = cv2.imread(dir+filename)
        object_instance["image"] = cv2.cvtColor(object_instance["image"], cv2.COLOR_BGR2RGB)
        
        mags, dirs = gradient_magnitudes(object_instance["image"], 2)
        angle_space = np.linspace(-np.pi, np.pi, 8) #in radians
        im_height_space = np.linspace(0, object_instance["image"].shape[0], 8)
        im_width_space = np.linspace(0, object_instance["image"].shape[1], 8)
        histograms = np.zeros((8,8,8))

        for y in range(object_instance["image"].shape[0]):
            for x in range(object_instance["image"].shape[1]):
                ybin = np.digitize(y, im_height_space) - 1
                xbin = np.digitize(x, im_width_space) - 1
                anglebin = np.digitize(dirs[y][x], angle_space)-1
                histograms[ybin][xbin][anglebin] += mags[y][x]

        object_instance["feature"] = histograms.flatten()

        objects_data[filename] = object_instance
    
    #return
    return objects_data

# %%
def compare_histograms(h1, h2, measure):
    """
        h1 : histogram 1
        h2 : histogram 2
        measure : chosen distance measure
    """
    result = -1

    if(measure=="l2"):
        result = np.sqrt(np.sum((h1-h2)**2))
    elif(measure=="chi-square"):
        result = 0.5 * np.sum(
            ((h1-h2)**2)
            /
            (h1+h2+1e-10)
        )
    elif(measure=="intersection"):
        result = 1 - np.sum(
            np.minimum(h1,h2)
        )
    elif(measure=="hellinger"):
        result = np.sqrt(
            0.5 * 
            np.sum(
                (np.sqrt(h1) - np.sqrt(h2))**2
            )
        )
    
    return result

# %%
histograms_data = img_retrieval("./dataset/", 8)

# %%
selected_image = "object_05_4.png" 

#init new dict with same keys - file names - but for holding distances to the selected image
distances_from_selected = dict.fromkeys(histograms_data)
#iterate over the file names and for each file compute all distances to the selected file
for filename in distances_from_selected:
    #init struct for distances for this instance
    distances_object = {
        "l2" : -1,
        "chi-square" : -1,
        "intersection" : -1,
        "hellinger" : -1
    }
    #this could be in a for loop over the method names, but i feel like this way is more readable and its already spaghetti
    distances_object["l2"] = compare_histograms(histograms_data[selected_image]["feature"], histograms_data[filename]["feature"], "l2")
    distances_object["chi-square"] = compare_histograms(histograms_data[selected_image]["feature"], histograms_data[filename]["feature"], "chi-square")
    distances_object["intersection"] = compare_histograms(histograms_data[selected_image]["feature"], histograms_data[filename]["feature"], "intersection")
    distances_object["hellinger"] = compare_histograms(histograms_data[selected_image]["feature"], histograms_data[filename]["feature"], "hellinger")
    #put it in the main struct
    distances_from_selected[filename] = distances_object




# %%
methods = ["hellinger"] #"l2", "chi-square", "intersection", 
sorted_by_distance = {}
for method in methods:
    #sort by distance from this method
    sorted_by_distance[method] = list(sorted(distances_from_selected.items(), key=lambda item: item[1][method]))
    #plot
    plt.figure(figsize=(18,6))
    ln_wid = 5
    for i in range(0,6):
        curr_name = sorted_by_distance[method][i][0]
        plt.subplot(2,6,i+1)
        plt.imshow(histograms_data[curr_name]["image"])
        plt.title(curr_name)
        curr_hist = histograms_data[curr_name]["feature"]
        plt.subplot(2,6,i+6+1)
        plt.bar(range(len(curr_hist)), curr_hist, width=ln_wid)
        plt.title("{} = {}".format(method, round(sorted_by_distance[method][i][1][method],2)))
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
def non_maxima_bad(mags, angls, thr):
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
def non_maxima(mags, angls):
    ret = np.pad(mags, pad_width=1, mode="constant", constant_values=0)
    
    for y in range(1, 1+mags.shape[0]):
        for x in range(1, 1+mags.shape[1]):
            angle = angls[y-1][x-1]
            angle = ((angle) * 180) / math.pi
            angle = angle + 180 if angle < 0 else angle
            crds = None
            if angle <= 22.5:
                crds = ((y, x-1), (y, x+1))
            elif angle <= 67.5:
                crds = ((y-1, x+1), (y-1, x+1))
            elif angle <= 112.5:
                crds = ((y-1, x), (y+1, x))
            elif angle <= 157.5:
                crds = ((y-1, x-1), (y+1, x+1))
            elif angle <= 180:
                crds = ((y, x-1), (y, x+1))
            if ret[crds[0][0]][crds[0][1]] > ret[y][x] or ret[crds[1][0]][crds[1][1]] > ret[y][x]:
                ret[y][x] = 0
    
    return ret
            

# %%
e2b_museum = cv2.imread("images\museum.jpg", cv2.IMREAD_GRAYSCALE)
e2b_museum = e2b_museum.astype(np.float64)

e2b_thr = 0.16
e2b_test1_mag, e2b_test1_dir = gradient_magnitudes(e2b_museum, 2)

#
e2b_test1_mag = (e2b_test1_mag - np.min(e2b_test1_mag)) / (np.max(e2b_test1_mag) - np.min(e2b_test1_mag))
e2b_test1_mag = np.where(e2b_test1_mag < e2b_thr, 0, e2b_test1_mag)

# e2b_test1_nms = non_maxima_bad(e2b_test1_mag, e2b_test1_dir, e2b_thr)

e2b_test1_nms2 = non_maxima(e2b_test1_mag, e2b_test1_dir)

plt.figure(figsize=(16,10))
plt.subplot(1,3,1)
plt.imshow(e2b_museum, cmap="gray")
plt.subplot(1,3,2)
plt.imshow(e2b_test1_mag, cmap="gray")
# plt.subplot(1,2,3)
# plt.imshow(e2b_test1_nms, cmap="gray")
plt.subplot(1,3,3)
plt.imshow(e2b_test1_nms2, cmap="gray")
plt.show()

# %% [markdown]
# #### (c)

# %%
def hysteresis(img, tlow, thigh):
    ret = np.copy(img)
    ret[ret>thigh] = 1
    ret[ret<tlow] = 0

    n_labs, labels, _, _ = cv2.connectedComponentsWithStats(np.uint8(ret*255), connectivity=8)

    for i in range(1,n_labs):
        label_mask = np.where(labels == i)
        ret[label_mask] = 1 if np.any(ret[label_mask] >= 1) else 0

    return ret

# %%
e3b_museum = e2a_museum
e3b_thr = 0.16
e3b_thigh = 0.16
e3b_tlow = 0.04

e3b_mag, e3b_dir = gradient_magnitudes(e3b_museum, 2)

e3b_thresh = np.where(e3b_mag < e3b_thr, 0, 1)
e3b_nms = non_maxima(e3b_mag, e3b_dir)
e3b_hist = hysteresis(e3b_nms, e3b_tlow, e3b_thigh)

# %%
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
plt.imshow(e3b_museum, cmap="gray")
plt.title("Original")
plt.subplot(2,2,2)
plt.imshow(e3b_thresh, cmap="gray")
plt.title("Thresholded")
plt.subplot(2,2,3)
plt.imshow(e3b_nms, cmap="gray")
plt.title("NMS")
plt.subplot(2,2,4)
plt.imshow(e3b_hist, cmap="gray")
plt.title("Hysteresis")
plt.show()

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

e3b_oneline = cv2.imread("./images/oneline.png", cv2.IMREAD_GRAYSCALE) / 255
e3b_rectangle = cv2.imread("./images/rectangle.png", cv2.IMREAD_GRAYSCALE) / 255
e3b_line_edges = findedges(e3b_oneline, 2, 0.5)
e3b_rect_edges = findedges(e3b_rectangle, 2, 0.5)
e3b_hough_2 = hough_find_lines(e3b_line_edges, (180, 180), None)
e3b_hough_3 = hough_find_lines(e3b_rect_edges, (180, 180), None)

plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
# a, b = gradient_magnitudes(np.uint8(e3b_line_edges*255), 2)
# plt.imshow(non_maxima(a, b), cmap="gray")
plt.imshow(e3b_line_edges, cmap="gray")
plt.subplot(1,2,2)
plt.imshow(e3b_rect_edges, cmap="gray")
plt.show()

# %%
plt.figure(figsize=(12,6))
plt.subplot(2,3,1)
plt.imshow(e3b_hough_1)
plt.title("Synthetic 100 x 100")
plt.subplot(2,3,2)
plt.imshow(e3b_hough_2)
plt.title("One line")
plt.subplot(2,3,3)
plt.imshow(e3b_hough_3)
plt.title("Rectangle")
# plt.subplot(2,3,4)
# plt.imshow(e3b_testimg_100, cmap="gray")
# plt.subplot(2,3,5)
# plt.imshow(e3b_line_edges, cmap="gray")
# plt.subplot(2,3,6)
# plt.imshow(e3b_rect_edges, cmap="gray")
plt.show()

# %% [markdown]
# #### (c)

# %%
def nonmaxima_suppression_box(matrx):
    ret = np.copy(matrx)
    ret = np.pad(matrx, pad_width=1, mode="constant", constant_values=0)
    
    for y in range(1, 1+matrx.shape[0]):
        for x in range(1, 1+matrx.shape[1]):
            coords = (
                (y-1, x-1),
                (y-1, x),
                (y-1, x+1),
                (y, x-1),
                (y, x+1),
                (y+1, x-1),
                (y+1, x),
                (y+1, x+1)
            )
            is_not_max = False
            for ny, nx in coords:
                if ret[ny][nx] > ret[y][x]:
                    is_not_max = True
                    break
            if is_not_max: ret[y][x] = 0
    
    return ret[1:-1,1:-1]

# %% [markdown]
# #### (d)

# %%
def pair_extraction(acc, img_dims, thr, bins):
    thrsh_acc = np.where(acc > thr, 1, 0)
    ret = []
    theta_bins, rho_bins = bins
    theta_space = np.linspace(-np.pi/2, np.pi/2, theta_bins)
    D = math.sqrt(img_dims[0]**2+img_dims[1]**2)
    rho_space = np.linspace(-D, D, rho_bins)

    for y in range(0, thrsh_acc.shape[0]):
        for x in range(0, thrsh_acc.shape[1]):
            if thrsh_acc[y][x] == 1: 
                theta = theta_space[x]
                rho = rho_space[y]
                ret.append((rho, theta))
    return ret

# %%
e3d_synt = e3b_testimg_100
e3d_line = e3b_oneline
e3d_rect = e3b_rectangle

e3d_synt_edges = e3d_synt
e3d_line_edges = findedges(e3d_line, 2, 0.2)
e3d_rect_edges = findedges(e3d_rect, 2, 0.2)

# e3d_synt_hough = hough_find_lines(e3d_synt_edges, (180, 180), None)
# e3d_line_hough = hough_find_lines(e3d_line_edges, (180, 180), None)
# e3d_rect_hough = hough_find_lines(e3d_rect_edges, (180, 180), None)

e3d_synt_hough = hough_find_lines(e3d_synt_edges, (180, 180), None)
e3d_line_hough = hough_find_lines(e3d_line_edges, (500, 500), None)
e3d_rect_hough = hough_find_lines(e3d_rect_edges, (750, 250), None)

e3d_synt_hgh_nms = nonmaxima_suppression_box(e3d_synt_hough)
e3d_line_hgh_nms = nonmaxima_suppression_box(e3d_line_hough)
e3d_rect_hgh_nms = nonmaxima_suppression_box(e3d_rect_hough)


# %%
e3d_pairs_synt = pair_extraction(e3d_synt_hgh_nms, (100, 100), 1, (180, 180))
e3d_pairs_line = pair_extraction(e3d_line_hgh_nms, (e3d_line.shape[0], e3d_line.shape[1]), 950, (500, 500))
e3d_pairs_rect = pair_extraction(e3d_rect_hgh_nms, (e3d_rect.shape[0], e3d_rect.shape[1]), 450, (750, 250))

plt.figure(figsize=(16,8))
plt.subplot(1,3,1)
plt.imshow(e3b_testimg_100, cmap="gray")
for rho, theta in e3d_pairs_synt:
    draw_line(rho, theta, 100, 100)
plt.subplot(1,3,2)
plt.imshow(e3b_oneline, cmap="gray")
for rho, theta in e3d_pairs_line:
    draw_line(rho, theta, e3b_oneline.shape[0], e3b_oneline.shape[1])
plt.subplot(1,3,3)
plt.imshow(e3b_rectangle, cmap="gray")
for rho, theta in e3d_pairs_rect:
    draw_line(rho, theta, e3b_rectangle.shape[0], e3b_rectangle.shape[1])

# draw_line(e3d_pairs_synt[0], e3d_pairs_synt[1], 100, 100)
# draw_line(e3d_pairs_line[0], e3d_pairs_line[1], e3b_oneline.shape[0], e3b_oneline.shape[1])
# draw_line(e3d_pairs_rect[0], e3d_pairs_rect[1], e3b_rectangle.shape[0], e3b_rectangle.shape[1])

# %% [markdown]
# #### (e)

# %%
def pair_extraction_topn(acc, img_dims, thr, bins, n):
    thrsh_acc = np.where(acc > thr, 1, 0)
    ret = []
    theta_bins, rho_bins = bins
    theta_space = np.linspace(-np.pi/2, np.pi/2, theta_bins)
    D = math.sqrt(img_dims[0]**2+img_dims[1]**2)
    rho_space = np.linspace(-D, D, rho_bins)

    for y in range(0, thrsh_acc.shape[0]):
        for x in range(0, thrsh_acc.shape[1]):
            if thrsh_acc[y][x] == 1: 
                theta = theta_space[x]
                rho = rho_space[y]
                ret.append((rho, theta, acc[y][x]))

    sort_ret = sorted(ret, key = lambda el: el[2], reverse=True)
    return sort_ret[0:n]

# %%
e3e_bricks = cv2.imread("./images/bricks.jpg", cv2.IMREAD_GRAYSCALE) / 255
e3e_pier = cv2.imread("./images/pier.jpg", cv2.IMREAD_GRAYSCALE) / 255

e3e_bricks_edges = findedges(e3e_bricks, 2, 0.1)
e3e_pier_edges = findedges(e3e_pier, 2, 0.2)

e3e_bricks_binsizes = (700, 700)
e3e_pier_binsizes   = (700, 700)

e3e_bricks_hough = hough_find_lines(e3e_bricks_edges, e3e_bricks_binsizes, None)
e3e_pier_hough = hough_find_lines(e3e_pier_edges, e3e_pier_binsizes, None)

e3e_bricks_nms = nonmaxima_suppression_box(e3e_bricks_hough)
e3e_pier_nms = nonmaxima_suppression_box(e3e_pier_hough)

# %%
e3e_bricks_pairs = pair_extraction_topn(e3e_bricks_nms, (e3e_bricks.shape[0], e3e_bricks.shape[1]), 1, e3e_bricks_binsizes, 10)
e3e_pier_pairs = pair_extraction_topn(e3e_pier_nms, (e3e_pier.shape[0], e3e_pier.shape[1]), 1, e3e_pier_binsizes, 10)

e3e_bricks_color = cv2.imread("./images/bricks.jpg")
e3e_bricks_color = cv2.cvtColor(e3e_bricks_color, cv2.COLOR_BGR2RGB)
e3e_pier_color = cv2.imread("./images/pier.jpg")
e3e_pier_color = cv2.cvtColor(e3e_pier_color, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
plt.imshow(e3e_bricks_hough)
plt.title("bricks.jpg")
plt.subplot(2,2,2)
plt.imshow(e3e_pier_hough)
plt.title("pier.jpg")
plt.subplot(2,2,3)
plt.imshow(e3e_bricks_color)
for rho, theta, _ in e3e_bricks_pairs:
    draw_line(rho, theta, e3e_bricks.shape[0], e3e_bricks.shape[1])
plt.subplot(2,2,4)
plt.imshow(e3e_pier_color)
for rho, theta, _ in e3e_pier_pairs:
    draw_line(rho, theta, e3e_pier.shape[0], e3e_pier.shape[1])
plt.show()


