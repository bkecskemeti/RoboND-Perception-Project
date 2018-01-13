import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from pcl_helper import *


def rgb_to_hsv(rgb_list):
    rgb_normalized = [1.0*rgb_list[0]/255, 1.0*rgb_list[1]/255, 1.0*rgb_list[2]/255]
    hsv_normalized = matplotlib.colors.rgb_to_hsv([[rgb_normalized]])[0][0]
    return hsv_normalized

def histogram(data, channels, nbins, bins_range):
    # Calculate histogram for each channel
    hists = [np.histogram([point[channel] for point in data], bins=nbins, range=bins_range)[0] for channel in channels]
    # Concatenate histograms for each channel into a single feature vector
    features = np.concatenate(tuple(hists)).astype(np.float64)
    # Normalize result
    return features / np.sum(features)

def compute_color_histograms(cloud, using_hsv=False):
    # Compute histograms for the clusters
    point_colors_list = []

    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    return histogram(point_colors_list, range(3), nbins=32, bins_range=(0, 256))


def compute_normal_histograms(normal_cloud):
    # Convert point cloud to array
    normals = pc2.read_points(normal_cloud,
                              field_names=('normal_x', 'normal_y', 'normal_z'),
                              skip_nans=True)

    return histogram(normals, range(3), nbins=32, bins_range=(-1, 1))

