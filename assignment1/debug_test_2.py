
import cv2

import numpy as np




def otsu(image, n_bins):
    """
    params
    ----------------------------------------------------------------------------------
     image : grayscale image 
     n_bins : how many bins to use for computing histograms 
    ----------------------------------------------------------------------------------
    """
    
    #init
    T = 0 #threshhold
    #max_t_val = 1 * n_bins
    max_var = 0 #max variance

    #histogram
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

    
    #iterate over possible threshold values
    for t in range(0,255):
        #potential_t_val = i / n_bins

        prob_of_bg = histogram[:t].sum()/image.size #probability of a pixel being in the background
        prob_of_fg = 1 - prob_of_bg #probability of a pixel being in the foreground
        
        if(prob_of_bg == 0 or prob_of_fg == 0): continue #skip iteration (current threshhold) if one class is empty

        mean_val_bg = (np.arange(0,t) * histogram[:t]).sum() / prob_of_bg #mean value of background class
        mean_val_fg = (np.arange(t,255+1) * histogram[t:]).sum() / prob_of_fg #mean value of foreground class

        bet_class_var = prob_of_bg * prob_of_fg * (mean_val_bg - mean_val_fg)**2 #between class variance

        if(bet_class_var > max_var):
            #if true we have a new best candidate for threshhold
            max_var = bet_class_var
            T = t
    

    return T

image = cv2.imread('C:/Users/Ziga/OneDrive - Univerza v Ljubljani/3. letnik/UZ/labs/assignment1/images/bird.jpg', 0)
print(otsu(image,100))