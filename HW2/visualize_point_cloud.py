import numpy as np
import matplotlib
# matplotlib.use('Qt4Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns


def visualizing_point_clouds(point_cloud,rgb):
    """
    Visualize a 3D point cloud using matplotlib.

    Args:
        point_cloud (numpy.ndarray): 3D point cloud represented as a NumPy array of shape (N, 3),
            where N is the number of points.

            rgb (numpy.ndarray): RGB color values corresponding to the 3D points, represented as a
            NumPy array of shape (N, 3), where N is the number of points.

    Returns:
        None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c=rgb)
    
    # make the plot interactive
    # plt.ion()
    plt.show()
    return fig


point_cloud_path = "/Users/sanyam/Projects/UMich/3D-robot-perception/HW2/HW2/point_cloud.npy"
rgb_path = "/Users/sanyam/Projects/UMich/3D-robot-perception/HW2/HW2/rgb.npy"

point_cloud = np.load(point_cloud_path)
rgb = np.load(rgb_path)

# plot every 10th point
point_cloud = point_cloud[::10]
rgb = rgb[::10]
visualizing_point_clouds(point_cloud,rgb)
