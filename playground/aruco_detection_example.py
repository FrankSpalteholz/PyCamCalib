import numpy as np
import cv2, PIL
from cv2 import aruco
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib as mpl
import pandas as pd

PI = 3.14159265359

def rad_to_deg(rad):
    return rad * (180/PI)

# Checks if a matrix is a valid rotation matrix.
def is_rotation_matrix(R,det_thresh) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    #print("n:" + str(n))
    return n < det_thresh

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotation_matrix_to_euler_angles(R, det_thresh) :
    assert(is_rotation_matrix(R,det_thresh))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

marker_length = 0.078 #in meters
draw_axis_length = 0.05
img_size = np.array([4032.0, 3024.0])
#img_size = np.array([1920.0, 1080.0])

tvec_avrg = np.zeros((1,1,3))
rvec_avrg = np.zeros((1,1,3))
rot_mat_avrg = np.zeros((3,3))

# 4032 × 3024 pixels
calib_photo = "calib_photo.0012.jpg"

matrix_coefficients = np.array([   [  2000.0,           0.00000000e+00,   img_size[0]/2],
                                   [  0.00000000e+00,   2000.0,           img_size[1]/2],
                                   [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

distortion_coefficients = np.zeros((5,1))

frame = cv2.imread(calib_photo)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_50)

parameters = aruco.DetectorParameters_create()

corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters,
                                                        rejectedImgPoints=None,
                                                        cameraMatrix=matrix_coefficients,
                                                        distCoeff=distortion_coefficients)
aruco.drawDetectedMarkers(frame, corners, ids)

for i in range(4):
    rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], marker_length,
                                                               matrix_coefficients,
                                                               distortion_coefficients)

    aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, draw_axis_length/2)  # Draw axis

    tvec_avrg += tvec
    rot_mat, _ = cv2.Rodrigues(rvec)
    rot_mat_avrg += rot_mat

tvec_avrg /= 4
rot_mat_avrg /= 4
rvec_avrg, _ = cv2.Rodrigues(rot_mat_avrg)

print("tvec_avrg: " + str(tvec_avrg) + " " + str(tvec_avrg.shape))
print("mag tvec_avrg: " + str(np.linalg.norm(tvec_avrg)))
print("rvec_avrg: " + str(rvec_avrg) + " " + str(rvec_avrg.shape))
print("rot_mat_avrg: " + str(rot_mat_avrg) + " " + str(rot_mat_avrg.shape))
print("det rot_mat_avrg: " + str(np.linalg.det(rot_mat_avrg)))

euler_angles = rotation_matrix_to_euler_angles(rot_mat_avrg, 1e-2)

print("euler angles rad: " + str(euler_angles[0]) + ":" + str(euler_angles[1]) + ":" + str(euler_angles[2]))
print("euler angles deg: " + str(rad_to_deg(euler_angles[0])) + ":" + str(rad_to_deg(euler_angles[1])) + ":" + str(rad_to_deg(euler_angles[2])))

aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec_avrg, tvec_avrg, draw_axis_length)  # Draw axis

cv2.imshow('image',frame)
cv2.waitKey(0)

