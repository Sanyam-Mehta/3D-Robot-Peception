import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import helper_functions as _helper




def compute_fundamental_matrix(pts1, pts2, scale):
    """
    Compute the Fundamental matrix from corresponding 2D points in two images.

    Given two sets of corresponding 2D image points from Image 1 (pts1) and Image 2 (pts2),
    as well as a scaling factor (scale) representing the maximum dimension of the images, 
    this function calculates the Fundamental matrix.

    Parameters:
    pts1 (numpy.ndarray): An Nx2 array containing 2D points from Image 1.
    pts2 (numpy.ndarray): An Nx2 array containing 2D points from Image 2, corresponding to pts1.
    scale (float): The maximum dimension of the images, used for scaling the Fundamental matrix.

    Returns:
    F (numpy.ndarray): A 3x3 Fundamental matrix 
    """
    F = None
    ####################################
    ##########YOUR CODE HERE############
    ####################################
    
    # Epipolar Constraint: https://www.youtube.com/watch?v=6kpBqfgSPRc&list=PL2zRqk16wsdoCCLpou-dGo7QQNks1Ppzo&index=8
    # Constraint Equation: w1^T * E * w2 = 0
    # w2 = [x2, y2, z2], w1 = [x1, y1, z1]
    # E = Essential Matrix = Skew_Symmetric(t) * R
    # But we only have 2D points, so we need to calculate the Fundamental Matrix
    # Solving the for w2 and w1, we get the equation: x1.T * F * x2 = 0
    # F = K1.inv().T * E * K2
    # K1, K2 are the intrinsic matrices of the two cameras [KNOWN!]
    # This is the epipolar constraint equation. 
    # We are trying to find the Fundamental matrix that satisfies this equation for all the points.
    # We need at least 8 points to solve for the Fundamental matrix.
    # Fundamental matrix constraint: norm should be 1.
    
    
    # How to estimate Fundamental matrix: https://www.youtube.com/watch?v=izpYAwJ0Hlw&list=PL2zRqk16wsdoCCLpou-dGo7QQNks1Ppzo&index=10
    
    # Check the pseudocode here: https://sites.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/jhan320/index.html#:~:text=Therefore%2C%20the%20fundamental%20matrix%20must,%2C%20resulting%20in%20U%20%CE%A3V'.
    # Constructing the A matrix (watch the video above)
    
    pts1 = pts1 / scale
    pts2 = pts2 / scale
    
    A = np.zeros((pts1.shape[0], 9))
    for i in range(pts1.shape[0]):
        A[i] = [
            pts1[i, 0]*pts2[i, 0], pts1[i, 0]*pts2[i, 1], pts1[i, 0], pts1[i, 1]*pts2[i, 0], pts1[i, 1]*pts2[i, 1], pts1[i, 1], pts2[i, 0], pts2[i, 1], 1
        ]
        
    # Solve for the Fundamental matrix. Least squares solution. Last column of V is the solution.
    
    _, _, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # Fundamental matrix is rank 2. So, we need to enforce this constraint.
    U_f, S_f, V_f = np.linalg.svd(F)
    
    S_f[2] = 0
    
    F = np.dot(U_f, np.dot(np.diag(S_f), V_f))
    
    ####################################
    return F 

def compute_epipolar_correspondences(img1, img2, pts1, F):
    """
    Compute epipolar correspondences in Image 2 for a set of points in Image 1 using the Fundamental matrix.

    Given two images (img1 and img2), a set of 2D points (pts1) in Image 1, and the Fundamental matrix (F)
    that relates the two images, this function calculates the corresponding 2D points (pts2) in Image 2.
    The computed pts2 are the epipolar correspondences for the input pts1.

    Parameters:
    img1 (numpy.ndarray): The first image containing the points in pts1.
    img2 (numpy.ndarray): The second image for which epipolar correspondences will be computed.
    pts1 (numpy.ndarray): An Nx2 array of 2D points in Image 1.
    F (numpy.ndarray): The 3x3 Fundamental matrix that relates img1 and img2.

    Returns:
    pts2_ep (numpy.ndarray): An Nx2 array of corresponding 2D points (pts2) in Image 2, serving as epipolar correspondences
                   to the points in Image 1 (pts1).
    """
    pts2_ep = None
    ####################################
    ##########YOUR CODE HERE############
    ####################################
    
    # To find dense correspondences, we need to find the epipolar lines in the second image.
    # Watch this video: https://www.youtube.com/watch?v=erpiFudDBlg&list=PL2zRqk16wsdoCCLpou-dGo7QQNks1Ppzo&index=11
    
    # The epipolar line in the second image is given by: l2 = F * x1 (We need the fundamental matrix to find the epipolar line)
    # Given point x1 in image 1 and the fundamental matrix F, we can find the epipolar line in the second image.
    # The corresponding point x2 in the second image should lie on this line.
    
    # epipolar line equation is given Lx2 = 0, where L = F * x1
    # Point x2 should lie on this line. All points on this line are the epipolar correspondences of x1.
    
    # To find the dense match, perform a 1-D search along the epipolar line, and find
    # minimum( img1[x1] - img2[x2] )^2
    
    # Construct the epipolar lines
    
    pts2_ep = np.zeros((pts1.shape[0], 2))
    
    for i in range(pts1.shape[0]):
        x1 = np.array([pts1[i, 0], pts1[i, 1], 1])
        l2 = np.dot(F, x1)
        
        # for all points that are on the line l2, and within the image bounds, find the best match
        # along the line.
        
        best_x2 = None
        best_y2 = None
        
        best_square_diff = float('inf')
        
        for j in range(img2.shape[1]):
            x = j
            y = int(-l2[2] - l2[0]*x / l2[1])
            
            x = int(x)
            y = int(y)
            
            # check if x, y is within the image bounds
            if x >= 0 and x < img2.shape[1] and y >= 0 and y < img2.shape[0]:
                square_diff = (img1[int(pts1[i, 1]), int(pts1[i, 0])] - img2[y, x])**2
                if square_diff < best_square_diff:
                    best_square_diff = square_diff
                    best_x2 = x
                    best_y2 = y
                    
        pts2_ep[i] = [best_x2, best_y2]        
   
    ####################################
    return pts2_ep

def compute_essential_matrix(K1, K2, F):
    """
    Compute the Essential matrix from the intrinsic matrices and the Fundamental matrix.

    Given the intrinsic matrices of two cameras (K1 and K2) and the 3x3 Fundamental matrix (F) that relates
    the two camera views, this function calculates the Essential matrix (E).

    Parameters:
    K1 (numpy.ndarray): The 3x3 intrinsic matrix for Camera 1.
    K2 (numpy.ndarray): The 3x3 intrinsic matrix for Camera 2.
    F (numpy.ndarray): The 3x3 Fundamental matrix that relates Camera 1 and Camera 2.

    Returns:
    E (numpy.ndarray): The 3x3 Essential matrix (E) that encodes the essential geometric relationship between
                   the two cameras.

    """
    E = None
    ####################################
    ##########YOUR CODE HERE############
    ####################################
    # Epipolar Constraint: https://www.youtube.com/watch?v=6kpBqfgSPRc&list=PL2zRqk16wsdoCCLpou-dGo7QQNks1Ppzo&index=8
    # Constraint Equation: w1^T * E * w2 = 0
    # w2 = [x2, y2, z2], w1 = [x1, y1, z1]
    # E = Essential Matrix = Skew_Symmetric(t) * R
    # But we only have 2D points, so we need to calculate the Fundamental Matrix
    # Solving the for w2 and w1, we get the equation: x1.T * F * x2 = 0
    # F = K1.inv().T * E * K2
    # K1, K2 are the intrinsic matrices of the two cameras [KNOWN!]
    # This is the epipolar constraint equation. 
    # We are trying to find the Fundamental matrix that satisfies this equation for all the points.
    # We need at least 8 points to solve for the Fundamental matrix.
    # Fundamental matrix constraint: norm should be 1.
    
    # Decompose F to get E
    # F = K2.T.inv() * E * K1.inv()
    # Formula for E
    E = np.matmul(K2.T, np.matmul(F, K1))
    
    # Another way is to use k1 before k2. It depends on how you are solvig the equation.
    # E = np.matmul(K1.T, np.matmul(F, K2))
    
    ####################################
    return E 

def triangulate_points_my_version(E, pts1_ep, pts2_ep, K1, K2):
    """
    Triangulate 3D points from the Essential matrix and corresponding 2D points in two images.

    Given the Essential matrix (E) that encodes the essential geometric relationship between two cameras,
    a set of 2D points (pts1_ep) in Image 1, and their corresponding epipolar correspondences in Image 2
    (pts2_ep), this function calculates the 3D coordinates of the corresponding 3D points using triangulation.

    Extrinsic matrix for camera1 is assumed to be Identity. 
    Extrinsic matrix for camera2 can be found by cv2.decomposeEssentialMat(). Note that it returns 2 Rotation and 
    one Translation matrix that can form 4 extrinsic matrices. Choose the one with the most number of points in front of 
    the camera.

    Parameters:
    E (numpy.ndarray): The 3x3 Essential matrix that relates two camera views.
    pts1_ep (numpy.ndarray): An Nx2 array of 2D points in Image 1.
    pts2_ep (numpy.ndarray): An Nx2 array of 2D points in Image 2, corresponding to pts1_ep.

    Returns:
    point_cloud (numpy.ndarray): An Nx3 array representing the 3D point cloud, where each row contains the 3D coordinates
                   of a triangulated point.
    point_cloud_cv (numpy.ndarray): An Nx3 array representing the 3D point cloud, where each row contains the 3D coordinates
                   of a triangulated point calculated using cv2.triangulate
    """
    
    # Refer this video https://www.youtube.com/watch?v=OYwm4VM6uNg&list=PL2zRqk16wsdoCCLpou-dGo7QQNks1Ppzo&index=12
    
    point_cloud = None
    point_cloud_cv = None
    ####################################
    ##########YOUR CODE HERE############
    ####################################
    
    # construct the projection matrices
    
    # Step 1. From the Essential matrix, we can get the Rotation and Translation matrices.
    extrinsic_matrix1 = np.eye(3, 4)
    
    R1, R2, t = cv2.decomposeEssentialMat(E)
    
    # Construct the 4 possible extrinsic matrices
    extrinsic_matrix2 = np.zeros((3, 4, 4))
    
    extrinsic_matrix2[:, :, 0] = np.hstack((R1, t))
    extrinsic_matrix2[:, :, 1] = np.hstack((R1, -t))
    extrinsic_matrix2[:, :, 2] = np.hstack((R2, t))
    extrinsic_matrix2[:, :, 3] = np.hstack((R2, -t))
    
    
    # Step 2. Construct the projection matrices
    projection_matrix1 = np.dot(K1, extrinsic_matrix1)
    
    projection_matrix2 = np.zeros((3, 4, 4))
    
    for i in range(4):
        projection_matrix2[:, :, i] = np.dot(K2, extrinsic_matrix2[:, :, i])
        
    # Step 3. Triangulate the points using all the projection matrices, and choose the one with the most points in front of the camera.

    point_cloud = np.zeros((pts1_ep.shape[0], 3))
    point_cloud_cv = np.zeros((pts1_ep.shape[0], 3))
    
    for i in range(pts1_ep.shape[0]):
        point1 = pts1_ep[i]
        point2 = pts2_ep[i]
        
        # Triangulate the points
        point1 = np.hstack((point1, 1))
        point2 = np.hstack((point2, 1))
        
        for j in range(4):
            # point_cloud_temp = cv2.triangulatePoints(projection_matrix1, projection_matrix2[:, :, j], point1, point2)
            # point_cloud_temp = point_cloud_temp / point_cloud_temp[3]
            
            # # Check if the point is in front of the camera
            # if point_cloud_temp[2] > 0:
            #     point_cloud[i] = point_cloud_temp[:3]
            #     point_cloud_cv[i] = point_cloud_temp[:3]
            #     break
            
            # Left camera projection matrix is K1 since the extrinsic matrix is identity
            M_intr = projection_matrix1 #K1 @ np.eye(3, 4)
            
            # Right camera projection matrix
            P_r = projection_matrix2[:, :, j]
            
            # Triangulate the points
            # u_l = M_intr @ X_r
            # u_r = P_r @ X_r
            
            # Construct the A matrix
            A = np.zeros((4, 4))
            A[0] = point1[0] * M_intr[2] - M_intr[0]
            A[1] = point1[1] * M_intr[2] - M_intr[1]
            A[2] = point2[0] * P_r[2] - P_r[0]
            A[3] = point2[1] * P_r[2] - P_r[1]
            
            b_l = np.zeros((4, 1))
            b_l[0] = M_intr[0, 3] - M_intr[2, 3] # m14 - m34
            b_l[1] = M_intr[1, 3] - M_intr[2, 3] # m24 - m34
            b_l[2] = P_r[0, 3] - P_r[2, 3] # p14 - p34
            b_l[3] = P_r[1, 3] - P_r[2, 3] # p24 - p34
            
            A = A[:, :3] # A is 4x3
            
            # Solve for A*x_l = b_l
            x_l = np.linalg.lstsq(A, b_l, rcond=None)[0]
            
            # Calculate the 3D point
            point_cloud_temp = x_l[:3] / x_l[3]
            
            # Check if the point is in front of the camera
            if point_cloud_temp[2] > 0:
                point_cloud[i] = point_cloud_temp
                point_cloud_cv[i] = point_cloud_temp
                break
            
    

    ####################################
    return point_cloud, point_cloud_cv


def triangulate_points(E, pts1_ep, pts2_ep, K1, K2):
    """
    Triangulate 3D points from the Essential matrix and corresponding 2D points in two images.

    Given the Essential matrix (E) that encodes the essential geometric relationship between two cameras,
    a set of 2D points (pts1_ep) in Image 1, and their corresponding epipolar correspondences in Image 2
    (pts2_ep), this function calculates the 3D coordinates of the corresponding 3D points using triangulation.

    Extrinsic matrix for camera1 is assumed to be Identity. 
    Extrinsic matrix for camera2 can be found by cv2.decomposeEssentialMat(). Note that it returns 2 Rotation and 
    one Translation matrix that can form 4 extrinsic matrices. Choose the one with the most number of points in front of 
    the camera.

    Parameters:
    E (numpy.ndarray): The 3x3 Essential matrix that relates two camera views.
    pts1_ep (numpy.ndarray): An Nx2 array of 2D points in Image 1.
    pts2_ep (numpy.ndarray): An Nx2 array of 2D points in Image 2, corresponding to pts1_ep.
    K1 (numpy.ndarray): The 3x3 intrinsic matrix for Camera 1.
    K2 (numpy.ndarray): The 3x3 intrinsic matrix for Camera 2.

    Returns:
    point_cloud (numpy.ndarray): An Nx3 array representing the 3D point cloud, where each row contains the 3D coordinates
                   of a triangulated point.
    point_cloud_cv (numpy.ndarray): An Nx3 array representing the 3D point cloud, where each row contains the 3D coordinates
                   of a triangulated point calculated using cv2.triangulate
    """
    
    point_cloud = np.zeros((pts1_ep.shape[0], 3))
    point_cloud_cv = np.zeros((pts1_ep.shape[0], 3))
    
    # Step 1. From the Essential matrix, we can get the Rotation and Translation matrices.
    extrinsic_matrix1 = np.eye(3, 4)
    
    R1, R2, t = cv2.decomposeEssentialMat(E)
    
    # Construct the 4 possible extrinsic matrices
    extrinsic_matrix2 = np.zeros((3, 4, 4))
    
    extrinsic_matrix2[:, :, 0] = np.hstack((R1, t))
    extrinsic_matrix2[:, :, 1] = np.hstack((R1, -t))
    extrinsic_matrix2[:, :, 2] = np.hstack((R2, t))
    extrinsic_matrix2[:, :, 3] = np.hstack((R2, -t))
    
    # Step 2. Construct the projection matrices
    projection_matrix1 = np.dot(K1, extrinsic_matrix1)
    
    projection_matrix2 = np.zeros((3, 4, 4))
    
    for i in range(4):
        projection_matrix2[:, :, i] = np.dot(K2, extrinsic_matrix2[:, :, i])
        
    # Step 3. Triangulate the points using all the projection matrices, and choose the one with the most points in front of the camera.
    
    for i in range(pts1_ep.shape[0]):
        point1 = pts1_ep[i]
        point2 = pts2_ep[i]
        
        # Triangulate the points
        point1 = np.hstack((point1, 1)).reshape(3, 1)
        point2 = np.hstack((point2, 1)).reshape(3, 1)
        
        for j in range(4):
            point_cloud_temp = cv2.triangulatePoints(projection_matrix1, projection_matrix2[:, :, j], point1[:2], point2[:2])
            point_cloud_temp = point_cloud_temp / point_cloud_temp[3]
            
            # Check if the point is in front of the camera
            if point_cloud_temp[2] > 0:
                point_cloud[i] = point_cloud_temp[:3].flatten()
                point_cloud_cv[i] = point_cloud_temp[:3].flatten()
                break
    
    return point_cloud, point_cloud_cv


def visualize(point_cloud, return_fig=False):
    """
    Function to visualize 3D point clouds
    Parameters:
    point_cloud (numpy.ndarray): An Nx3 array representing the 3D point cloud,where each row contains the 3D coordinates
                   of a triangulated point.
    """
    ####################################
    ##########YOUR CODE HERE############
    ####################################
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
    
    if return_fig:
        return fig, ax
    else:
        plt.show()
    
    ####################################


if __name__ == "__main__":
    data_for_fundamental_matrix = np.load("data/corresp_subset.npz")
    pts1_for_fundamental_matrix = data_for_fundamental_matrix['pts1']
    pts2_for_fundamental_matrix = data_for_fundamental_matrix['pts2']

    img1 = cv2.imread('data/im1.png')
    img2 = cv2.imread('data/im2.png')
    scale = max(img1.shape)
    

    data_for_temple = np.load("data/temple_coords.npz")
    pts1_epipolar = data_for_temple['pts1']

    data_for_intrinsics = np.load("data/intrinsics.npz")
    K1 = data_for_intrinsics['K1']
    K2 = data_for_intrinsics['K2']

    ####################################
    ##########YOUR CODE HERE############
    ####################################

    ####################################




