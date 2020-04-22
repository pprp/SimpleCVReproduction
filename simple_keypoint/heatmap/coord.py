import numpy as np
import matplotlib.pyplot as plt

input_height, input_width = 360, 480

"""
https://fairyonice.github.io/Achieving-top-5-in-Kaggles-facial-keypoints-detection-using-FCN.html#Model-performance-on-example-training-images
"""

def get_ave_xy(hmi, n_points=1, thresh=0):
    '''
    hmi      : heatmap np array of size (height,width)
    n_points : x,y coordinates corresponding to the top  densities to calculate average (x,y) coordinates


    convert heatmap to (x,y) coordinate
    x,y coordinates corresponding to the top  densities 
    are used to calculate weighted average of (x,y) coordinates
    the weights are used using heatmap

    if the heatmap does not contain the probability > 
    then we assume there is no predicted landmark, and 
    x = -1 and y = -1 are recorded as predicted landmark.
    '''
    if n_points < 1:
        # Use all
        hsum, n_points = np.sum(hmi), len(hmi.flatten())
        ind_hmi = np.array([range(input_width)]*input_height)
        i1 = np.sum(ind_hmi * hmi)/hsum
        ind_hmi = np.array([range(input_height)]*input_width).T
        i0 = np.sum(ind_hmi * hmi)/hsum
    else:
        ind = np.argsort(hmi,axis=None)[-n_points:]  
        # pick the largest n_points
        topind = np.unravel_index(ind, hmi.shape)
        index = np.unravel_index(hmi.argmax(), hmi.shape)
        i0, i1, hsum = 0, 0, 0
        for ind in zip(topind[0], topind[1]):
            h = hmi[ind[0], ind[1]]
            hsum += h
            i0 += ind[0]*h
            i1 += ind[1]*h

        i0 /= hsum
        i1 /= hsum
    if hsum/n_points <= thresh:
        i0, i1 = -1, -1
    return([i1, i0])


def transfer_xy_coord(hm, n_points=64, thresh=0.2):
    '''
    hm : np.array of shape (height,width, n-heatmap)
    transfer heatmap to (x,y) coordinates
    the output contains np.array (Nlandmark * 2,) 
    * 2 for x and y coordinates, containing the landmark location.
    '''
    assert len(hm.shape) == 3
    Nlandmark = hm.shape[-1]
    #est_xy = -1*np.ones(shape = (Nlandmark, 2))
    est_xy = []
    for i in range(Nlandmark):
        hmi = hm[:, :, i]
        est_xy.extend(get_ave_xy(hmi, n_points, thresh))
    return(est_xy)  # (Nlandmark * 2,)


def transfer_target(y_pred, thresh=0, n_points=1):
    '''
    y_pred : np.array of the shape (N, height, width, Nlandmark)
    output : (N, Nlandmark * 2)
    '''
    y_pred_xy = []
    for i in range(y_pred.shape[0]):
        hm = y_pred[i]
        y_pred_xy.append(transfer_xy_coord(hm, n_points, thresh))
    return(np.array(y_pred_xy))
