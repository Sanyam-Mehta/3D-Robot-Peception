import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import helper_functions as _helper
from camera_calibration import calculate_reprojection_error




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
    
    # Study scaling strategy here: https://www5.cs.fau.de/fileadmin/lectures/2014s/Lecture.2014s.IMIP/exercises/4/exercise4.pdf
    
    # center_1 = np.mean(pts1, axis=0)
    # center_2 = np.mean(pts2, axis=0)
    
    # original_pts1 = pts1.copy()
    # original_pts2 = pts2.copy()
    
    # T1 = np.array([
    #     [1/scale, 0, -center_1[0]/scale],
    #     [0, 1/scale, -center_1[1]/scale],
    #     [0, 0, 1]
    # ])
    
    # T2 = np.array([
    #     [1/scale, 0, -center_2[0]/scale],
    #     [0, 1/scale, -center_2[1]/scale],
    #     [0, 0, 1]
    # ])
    
    # # pts1 = T1 * pts1
    # # pts2 = T2 * pts2
    # pts1 = np.dot(T1, np.hstack((pts1, np.ones((pts1.shape[0], 1)))).T).T[:, :2]
    # pts2 = np.dot(T2, np.hstack((pts2, np.ones((pts2.shape[0], 1)))).T).T[:, :2]
    
    # A = np.zeros((pts1.shape[0], 9))
    # for i in range(pts1.shape[0]):
    #    A[i] = [
    #         pts1[i, 0] * pts2[i, 0],  # Points_a(i, 1) * Points_b(i, 1)
    #         pts1[i, 1] * pts2[i, 0],  # Points_a(i, 2) * Points_b(i, 1)
    #         pts2[i, 0],               # Points_b(i, 1)
    #         pts1[i, 0] * pts2[i, 1],  # Points_a(i, 1) * Points_b(i, 2)
    #         pts1[i, 1] * pts2[i, 1],  # Points_a(i, 2) * Points_b(i, 2)
    #         pts2[i, 1],               # Points_b(i, 2)
    #         pts1[i, 0],               # Points_a(i, 1)
    #         pts1[i, 1],               # Points_a(i, 2)
    #         1                         # 1
    # ]
        
    # # Solve for the Fundamental matrix. Least squares solution. Last column of V is the solution.
    
    # _, _, V = np.linalg.svd(A)
    # F = V[-1].reshape(3, 3)

    # # Fundamental matrix is rank 2. So, we need to enforce this constraint.
    # U_f, S_f, V_f = np.linalg.svd(F)
    
    # S_f[2] = 0
    
    # F = np.dot(U_f, np.dot(np.diag(S_f), V_f))
    
    
    # # Denormalize the Fundamental matrix
    # F = np.dot(T2.T, np.dot(F, T1))
    ####################################
    
    # Above translation not working for some reason. So, using the below code.
    pts1 = pts1 / scale
    pts2 = pts2 / scale

    # Construct matrix A
    A = np.zeros((pts1.shape[0], 9))
    for i in range(pts1.shape[0]):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A[i] = [
            x1 * x2, y1 * x2, x2,
            x1 * y2, y1 * y2, y2,
            x1, y1, 1
        ]

    # Solve for F using SVD
    _, _, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # Enforce rank-2 constraint
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt

    # Denormalize F using scaling matrices
    T = np.array([[1/scale, 0, 0],
                  [0, 1/scale, 0],
                  [0, 0, 1]])
    F = T.T @ F @ T  # Denormalize
    
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
    
    ## Personal Notes:
    # 1. Compute the epipolar lines in the second image. It is F @ x1
    # 2. For each column idx in the second image, find the y coordinate of the point on the line in the second image.
    # 3. Check if the point is within the image bounds.
    # 4. If it is within the image bounds, take a small window around the point and find the best match.
    # 5. The best match is the point that has the minimum squared difference with the point in the first image.
    
    
    pts2_ep = np.zeros((pts1.shape[0], 2))
    
    for i in range(pts1.shape[0]):
        x1 = np.array([pts1[i, 0], pts1[i, 1], 1])
        # l2 is F.T * x1
        l2 = F @ x1 #np.dot(F, x1)
        s = np.sqrt(l2[0]**2 + l2[1]**2)
        # import pdb; pdb.set_trace()
        if s == 0:
            print('Zero line vector in Epipolar lines')
            continue
        # l2 = l2 / s

        
        # for all points that are on the line l2, and within the image bounds, find the best match
        # along the line.
        
        best_x2 = None
        best_y2 = None
        
        best_square_diff = float('inf')
        
        for j in range(img2.shape[1]):
            x = j
            if l2[1] != 0:
                y = -(l2[0] * x + l2[2]) / l2[1]
            else:
                y = 0
            
            x = int(x)
            y = int(y)
            # import pdb; pdb.set_trace()
            # check if x, y is within the image bounds
            if x >= 0 and x < img2.shape[1] and y >= 0 and y < img2.shape[0]:
                # take a small window around x, y and find the best match
                window_size = 5
                # check if the window is within the image bounds
                if x - window_size < 0 or x + window_size >= img2.shape[1] or y - window_size < 0 or y + window_size >= img2.shape[0]:
                    continue
                    window_size = min(x, img2.shape[1] - x, y, img2.shape[0] - y)
                    
                # # find the best match in the window
                # square_diff = np.sum((img1[pts1[i, 1] - window_size:pts1[i, 1] + window_size, pts1[i, 0] - window_size:pts1[i, 0] + window_size] - img2[y - window_size:y + window_size, x - window_size:x + window_size])**2)
                
                # compute manhattan distance for window_size in both images
                image_1_window = img1[pts1[i, 1] - window_size:pts1[i, 1] + window_size, pts1[i, 0] - window_size:pts1[i, 0] + window_size]
                image_2_window = img2[y - window_size:y + window_size, x - window_size:x + window_size]
                
                diff = np.abs((np.array(image_2_window).astype(np.float32) -  np.array(image_1_window).astype(np.float32)))
                
                manhattan_distance = np.sum(diff)
                
                
                distance_between_coords = np.abs(x - pts1[i, 1]) + np.abs(y - pts1[i, 0])
                
                square_diff = manhattan_distance #+ distance_between_coords
                
                # import pdb; pdb.set_trace()
                if square_diff < best_square_diff:
                    best_square_diff = square_diff
                    best_x2 = x
                    best_y2 = y
         
        assert best_x2 is not None           
        pts2_ep[i] = [best_x2, best_y2]        
   
    # import pdb; pdb.set_trace()
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
    # E = np.matmul(K2.T, np.matmul(F, K1))
    
    # Another way is to use k1 before k2. It depends on how you are solvig the equation.
    E = np.matmul(K1.T, np.matmul(F, K2))
    
    ####################################
    return E 

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
    
    # Step 1. Use the Essential matrix to get the Rotation and Translation matrices.
    U_e, S_e, VT_e = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = np.dot(U_e, np.dot(W, VT_e))
    R2 = np.dot(U_e, np.dot(W.T, VT_e))
    
    t1 = U_e[:, 2]
    t2 = -U_e[:, 2]
    
    
    # Step 2. Construct 4 possible extrinsic matrices for camera 2
    possible_extrinsic_matrices = [
        np.hstack((R1, t1.reshape(3, 1))),
        np.hstack((R1, t2.reshape(3, 1))),
        np.hstack((R2, t1.reshape(3, 1))),
        np.hstack((R2, t2.reshape(3, 1)))
    ]
    
    # for each of these extrinsic matrices, construct the projection matrix for camera 2
    camera_2_projection_matrices = []
    for extrinsic_matrix in possible_extrinsic_matrices:
        camera_2_projection_matrices.append(np.dot(K2, extrinsic_matrix))
        
    camera_1_projection_matrix = np.dot(K1, np.hstack((np.eye(3), np.zeros((3, 1)))))
    
    # Solve Ax = 0, where A is the matrix of the form:
    # Row 1: pts1_ep[0, 0] * P1[2, :] - P1[0, :]
    # Row 2: pts1_ep[0, 1] * P1[2, :] - P1[1, :]
    # Row 3: pts2_ep[0, 0] * P2[2, :] - P2[0, :]
    # Row 4: pts2_ep[0, 1] * P2[2, :] - P2[1, :]
    # x is the 3D point in real world coordinates
    
    computed_3d_points = {
        0: [],
        1: [],
        2: [],
        3: [],
    }
    
    points_clouds = {
        0: [],
        1: [],
        2: [],
        3: [],
    }
    
    
    for projection_idx, P2 in enumerate(camera_2_projection_matrices):
        for i in range(pts1_ep.shape[0]):
            # compute the A matrix for each point, and solve for x
            # x is the 3D point in real world coordinates
            A = np.zeros((4, 4))
            A[0] = pts1_ep[i, 0] * camera_1_projection_matrix[2, :] - camera_1_projection_matrix[0, :]
            A[1] = pts1_ep[i, 1] * camera_1_projection_matrix[2, :] - camera_1_projection_matrix[1, :]
            A[2] = pts2_ep[i, 0] * P2[2, :] - P2[0, :]
            A[3] = pts2_ep[i, 1] * P2[2, :] - P2[1, :]
            
            # import pdb; pdb.set_trace()
            
            try:
                _, _, V = np.linalg.svd(A)
            except np.linalg.LinAlgError:
                continue
            x = V[-1]
            
            x = x / x[3]
            
            # import pdb; pdb.set_trace()
            
            # Reproject the 3D point to image 1 and image 2, compute the reprojection error
            x2_reprojected = np.dot(P2, x)
            
            x2_reprojected = x2_reprojected / x2_reprojected[2]
            
            
            computed_3d_points[projection_idx].append((x, x2_reprojected[:2]))
            
            # Append the 3D point to the point cloud
            points_clouds[projection_idx].append(x)
            
    # import pdb; pdb.set_trace()
            
    # Compute the reprojection error for each of the 3D points
    best_error = float('inf')
    best_error_idx = None
    
    for key in computed_3d_points.keys():
        # the first element of the tuple is the 3D point, and the second element is the reprojected 2D point
        pts_2_reprojected = np.array([x[1] for x in computed_3d_points[key]])
        
        pts_2d = np.array(pts2_ep).astype(np.float32)
        # import pdb; pdb.set_trace()
        
        # import pdb; pdb.set_trace()
        error = np.linalg.norm(pts_2d - pts_2_reprojected, axis=1)
        
        error = np.mean(error)
        
        print("For Projection Matrix: ", key, " Error: ", error)
        
        # import pdb; pdb.set_trace()
        
        # error = np.linalg.norm(pts_2d - pts_2_predicted, axis=1)
        
        if error < best_error:
            best_error = error
            best_error_idx = key
            
    best_error = 2
    
    print("Extrinsic Matrix: ", possible_extrinsic_matrices[best_error_idx])
    
    print("Extrinsic Matrix 0: ", possible_extrinsic_matrices[0])
            
    point_cloud = np.array(points_clouds[best_error_idx])
    
    # Triangulate using cv2.triangulate
    # Modified CV2 triangulation and error calculation section
    point_cloud_cv = []
    cv2_reprojections = []
    
    import pdb; pdb.set_trace()
    cv2_point_cloud = cv2.triangulatePoints(camera_1_projection_matrix, 
                                              camera_2_projection_matrices[best_error_idx], 
                                              pts1_ep.T, pts2_ep.T)
    for i in range(pts1_ep.shape[0]):
        
        # Project the 3D point back to image 2
        point_3d_homog = cv2_point_cloud.T[i].reshape(4, 1)
        proj_2 = np.dot(camera_2_projection_matrices[best_error_idx], point_3d_homog)
        proj_2 = proj_2 / proj_2[2]  # Normalize by z-coordinate
        cv2_reprojections.append(proj_2[:2].flatten())
        
    # import pdb; pdb.set_trace()

    point_cloud_cv = cv2_point_cloud.T / cv2_point_cloud.T[3] #np.array(point_cloud_cv)
    cv2_reprojections = np.array(cv2_reprojections).astype(np.float32)

    # Calculate reprojection error
    error = np.mean(np.linalg.norm(pts2_ep - cv2_reprojections, axis=1))
    
    # error = np.mean(error)
    print("CV2 Triangulation Error: ", error)
        
    
                        
    point_cloud_cv = np.array(point_cloud_cv)
    point_cloud_cv = point_cloud_cv / point_cloud_cv[3]
    
    # point_cloud_cv = point_cloud_cv.T[:, :3]
    
    ####################################

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
    data_for_fundamental_matrix = np.load("UMich/3D-robot-perception/HW1/src/data/corresp_subset.npz")
    pts1_for_fundamental_matrix = data_for_fundamental_matrix['pts1']
    pts2_for_fundamental_matrix = data_for_fundamental_matrix['pts2']

    img1 = cv2.imread('UMich/3D-robot-perception/HW1/src/data/im1.png')
    img2 = cv2.imread('UMich/3D-robot-perception/HW1/src/data/im2.png')
    scale = max(img1.shape)
    

    data_for_temple = np.load("UMich/3D-robot-perception/HW1/src/data/temple_coords.npz")
    pts1_epipolar = data_for_temple['pts1']

    data_for_intrinsics = np.load("UMich/3D-robot-perception/HW1/src/data/intrinsics.npz")
    K1 = data_for_intrinsics['K1']
    K2 = data_for_intrinsics['K2']

    ####################################
    ##########YOUR CODE HERE############
    ####################################
    
    # get the fundamental matrix
    
    F = compute_fundamental_matrix(
        pts1_for_fundamental_matrix, 
        pts2_for_fundamental_matrix, 
        scale
    )
    # print(F)
    # compute fundamental matriz using cv2
    F_cv, _ = cv2.findFundamentalMat(pts1_for_fundamental_matrix, pts2_for_fundamental_matrix, cv2.FM_LMEDS)

    # import pdb; pdb.set_trace()
    
    # visualize the epipolar lines
    # _helper.epipolar_lines_GUI_tool(
    #     img1, 
    #     img2, 
    #     F
    # )
    
    # _helper.epipolar_correspondences_GUI_tool(
    #     img1, 
    #     img2, 
    #     F
    # )
    
    
    # # get the essential matrix
    E = compute_essential_matrix(
        K1, 
        K2, 
        F
    )
    
    E_cv = cv2.findEssentialMat(pts1_for_fundamental_matrix, pts2_for_fundamental_matrix, K1, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    E_cv = E_cv[0]
    
    print("Fundamental Matrix: ", F)
    print("Essential Matrix: ", E)
    
    # # get the epipolar correspondences
    pts2_ep = compute_epipolar_correspondences(
        img1, 
        img2, 
        pts1_epipolar, 
        F
    )
    import pdb; pdb.set_trace()
    
    # Ensure the points are in the correct format
    pts1_epipolar = np.array(pts1_epipolar, dtype=np.float32)
    
    # # # # triangulate the points
    # point_cloud, _ = triangulate_points(
    #     E, 
    #     pts1_epipolar, 
    #     pts2_ep, 
    #     K1, 
    #     K2
    # )
    
    point_cloud, point_cloud_cv = triangulate_points(
        E, 
        pts1_epipolar, 
        pts2_ep, 
        K1, 
        K2
    )
    
    import pdb; pdb.set_trace()
    
    # point_cloud, _ = triangulate_points(
    #     E, 
    #     pts1_epipolar, 
    #     pts2_ep, 
    #     K1, 
    #     K2
    # )
    
    import pdb; pdb.set_trace()
    
    # # # visualize the point cloud
    visualize(point_cloud)

    # ####################################




