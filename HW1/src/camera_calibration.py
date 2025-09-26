import cv2
import os
import sys
import numpy as np
from pathlib import Path

def calculate_projection(pts2d, pts3d, svd=True):
    """
    Compute a 3x4 projection matrix M using a set of 2D-3D point correspondences.

    Given a set of N 2D image points (pts2d) and their corresponding 3D world coordinates
    (pts3d), this function calculates the projection matrix M using the Direct Linear
    Transform (DLT) method. The projection matrix M relates the 3D world coordinates to
    their 2D image projections in homogeneous coordinates.

    Parameters:
    pts2d (numpy.ndarray): An Nx2 array containing the 2D image points.
    pts3d (numpy.ndarray): An Nx3 array containing the corresponding 3D world coordinates.

    Returns:
    M (numpy.ndarray): A 3x4 projection matrix M that relates 3D world coordinates to 2D
                   image points in homogeneous coordinates.
    """
    M = None
    ####################################
    ##########YOUR CODE HERE############
    ####################################
    
    # Following this video: https://www.youtube.com/watch?v=GUbWsXU1mac (Time Stamp)
    
    # Ap = 0
    
    # p is the 12*1 projection matrix vector, and A is 2N*12 matrix, N is the number of points
    # The even index of A solve for the equation that combines the 3D points to the u coordinate of the 2D points
    # The odd index of A solve for the equation that combines the 3D points to the v coordinate of the 2D points
    
    A = np.zeros((2*pts2d.shape[0], 12))
    
    
    # populating even rows [idx 0, 2, 4....]
    for i in range(0, 2*pts2d.shape[0], 2):
        A[i, 0] = pts3d[i//2, 0]
        A[i, 1] = pts3d[i//2, 1]
        A[i, 2] = pts3d[i//2, 2]
        A[i, 3] = 1
        A[i, 8] = -pts2d[i//2, 0]*pts3d[i//2, 0]
        A[i, 9] = -pts2d[i//2, 0]*pts3d[i//2, 1]
        A[i, 10] = -pts2d[i//2, 0]*pts3d[i//2, 2]
        A[i, 11] = -pts2d[i//2, 0]

    # populating odd rows [idx 1, 3, 5....]
    for i in range(1, 2*pts2d.shape[0], 2):
        A[i, 4] = pts3d[i//2, 0]
        A[i, 5] = pts3d[i//2, 1]
        A[i, 6] = pts3d[i//2, 2]
        A[i, 7] = 1
        A[i, 8] = -pts2d[i//2, 1]*pts3d[i//2, 0]
        A[i, 9] = -pts2d[i//2, 1]*pts3d[i//2, 1]
        A[i, 10] = -pts2d[i//2, 1]*pts3d[i//2, 2]
        A[i, 11] = -pts2d[i//2, 1]
        
    # Solve for the projection matrix
    _, _, V = np.linalg.svd(A)
    
    # The projection matrix is the last column of V
    M = V[-1].reshape(3, 4)
    
    
    # The rational behind using svd and the last column of V:
    # The least squares solutions to the equation Ap = 0 can be written as optimizing:
    
    # min ||Ap||^2 subject to ||p|| = 1
    # min (Ap)^T(Ap) subject to p^Tp = 1
    # min p^T A^T A p subject to p^Tp = 1
    # Solving this optimization problem (lagrange multiplier), we find that the solution of the problem
    # is a eignevector of A^T A with the smallest eigenvalue. To minimize, we choose the smallest 
    # eigenvalue of A^T A, which is the last column of V, that we can obtain using SVD of A.
    
    # Get least eigenvalue of A^T A
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(A.T, A))
    
    # assert that the last column of V is the eigenvector corresponding to the smallest eigenvalue
    # import pdb; pdb.set_trace()
    # assert np.allclose(V[-1], eigenvectors[:, np.argmin(eigenvalues)])
    
    if not svd:
        M = eigenvectors[:, np.argmin(eigenvalues)].reshape(3, 4)
    

    ####################################
    return M


def calculate_reprojection_error(pts2d,pts3d, svd=True):
    """
    Calculate the reprojection error for a set of 2D-3D point correspondences.

    Given a set of N 2D image points (pts2d) and their corresponding 3D world coordinates
    (pts3d), this function calculates the reprojection error. The reprojection error is a
    measure of how accurately the 3D points project onto the 2D image plane when using a
    projection matrix.

    Parameters:
    pts2d (numpy.ndarray): An Nx2 array containing the 2D image points.
    pts3d (numpy.ndarray): An Nx3 array containing the corresponding 3D world coordinates.

    Returns:
    float: The reprojection error, which quantifies the accuracy of the 3D points'
           projection onto the 2D image plane.
    """
    M = calculate_projection(pts2d, pts3d, svd)
    error = None
    ####################################
    ##########YOUR CODE HERE############
    ####################################
    
    # use the 3*4 projection matrix M to project the N * 4 3D world coordinates
    
    pts3d_homogeneous = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))
    # import pdb; pdb.set_trace()
    pts2d_predicted = np.dot(M, pts3d_homogeneous.T).T
    
    # last column of the predicted 2D points are the homogeneous coordinates, divide by the last column
    pts2d_predicted = pts2d_predicted / pts2d_predicted[:, 2].reshape(-1, 1)
    
    # Calculate the error
    pts2d_predicted = pts2d_predicted[:, :2]
    
    error = np.linalg.norm(pts2d - pts2d_predicted, axis=1)
    
    error = np.mean(error)
    
    ####################################
    return error


if __name__ == '__main__':
    
    file_path = Path(__file__).resolve()
    print(file_path.parent)
    data = np.load(file_path.parent / "data/camera_calib_data.npz")
    pts2d = data['pts2d']
    pts3d = data['pts3d']

    # Compute the projection matrix using SVD
    P = calculate_projection(pts2d,pts3d, svd=True)
    reprojection_error = calculate_reprojection_error(pts2d, pts3d)
    
    # Compute the projection matrix using Lagrange Multiplier
    P_lagrange = calculate_projection(pts2d,pts3d, svd=False)
    reprojection_error_lagrange = calculate_reprojection_error(pts2d, pts3d, svd=False)

    print(f"Projection matrix: {P}")    
    print()
    print(f"Reprojection Error: {reprojection_error}")
    
    print(f"Projection matrix using Lagrange Multiplier: {P_lagrange}")
    print()
    print(f"Reprojection Error using Lagrange Multiplier: {reprojection_error_lagrange}")
    
    
    # Assert that P and P_lagrange are the same (P_lagrange has opposite sign)
    assert np.allclose(P, -P_lagrange)