import numpy as np
import cv2
from cv2 import aruco
import math

def rad_to_deg(rad):
    return rad * (180/math.pi)

def is_rotation_matrix(R,det_thresh) :

    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    #print("n:" + str(n))
    return n < det_thresh

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

def output_extrinsics(tvec, rvec, rot_mat, matrix_coeffs, dist_coeffs, image_path):

    frame = cv2.imread(image_path)
    draw_axis_length = 0.05
    print("tvec_avrg: " + str(tvec) + " " + str(tvec.shape))
    print("mag tvec_avrg: " + str(np.linalg.norm(tvec)))
    print("rvec_avrg: " + str(rvec) + " " + str(rvec.shape))
    print("rot_mat_avrg: " + str(rot_mat) + " " + str(rot_mat.shape))
    print("det rot_mat_avrg: " + str(np.linalg.det(rot_mat)))

    euler_angles = rotation_matrix_to_euler_angles(rot_mat, 1e-2)

    print("euler angles rad: " + str(euler_angles[0]) + ":" + str(euler_angles[1]) + ":" + str(euler_angles[2]))
    print("euler angles deg: " + str(rad_to_deg(euler_angles[0])) + ":" + str(rad_to_deg(euler_angles[1])) + ":" + str(
        rad_to_deg(euler_angles[2])))

    aruco.drawAxis(frame, matrix_coeffs, dist_coeffs, rvec, tvec,
                   draw_axis_length)  # Draw axis

    cv2.imshow('image', frame)
    cv2.waitKey(0)

def calc_extrinsics_from_markers(image_path, matrix_coeffs, dist_coeffs, marker_len, marker_num):

    tvec_tmp = np.zeros((1, 1, 3))
    rot_mat_tmp = np.zeros((3, 3))

    frame = cv2.imread(image_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_50)
    parameters = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(gray,
                                          aruco_dict,
                                          parameters=parameters,
                                          rejectedImgPoints=None,
                                          cameraMatrix=matrix_coeffs,
                                          distCoeff=dist_coeffs)

    # aruco.drawDetectedMarkers(frame, corners, ids)

    for i in range(marker_num):
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i],
                                                        marker_len,
                                                        matrix_coeffs,
                                                        dist_coeffs)
        tvec_tmp += tvec
        rot_mat, _ = cv2.Rodrigues(rvec)
        rot_mat_tmp += rot_mat

    tvec_tmp /= marker_num
    rot_mat_tmp /= marker_num
    rvec_tmp, _ = cv2.Rodrigues(rot_mat_tmp)

    return tvec_tmp, rvec_tmp, rot_mat_tmp

########################################################################################################################

# 4032 × 3024 pixels
calib_photo = "calib_image.jpg"

img_size = np.array([4032.0, 3024.0])
#img_size = np.array([1920.0, 1080.0])

marker_length = 0.078 #in meters
marker_num = 4
intrinsic_coeffs = np.array([   [  2000.0,           0.00000000e+00,   img_size[0]/2],
                                [  0.00000000e+00,   2000.0,           img_size[1]/2],
                                [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

dist_coeffs = np.zeros((5,1))


tvec_origin, rvec_origin, rot_mat_origin = calc_extrinsics_from_markers(calib_photo,
                                                                        intrinsic_coeffs,
                                                                        dist_coeffs,
                                                                        marker_length,
                                                                        marker_num)
output_extrinsics(tvec_origin,
                  rvec_origin,
                  rot_mat_origin,
                  intrinsic_coeffs,
                  dist_coeffs,
                  calib_photo)


