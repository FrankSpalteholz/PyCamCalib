import os
import numpy as np
import cv2
from cv2 import aruco
import math
from tools.cv_file_io import cv_file_io

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

def output_extrinsics(tvec, rvec, rot_mat, matrix_coeffs, dist_coeffs, image_path, index):

    frame = cv2.imread(image_path)
    draw_axis_length = 0.1

    print("========================================================================")
    print("[Calc extrinsics camera: " + str(index+1) + "]\n")

    print("[Tvec: " + str(tvec) + " " + str(tvec.shape))
    print("[Rvec: " + str(rvec) + " " + str(rvec.shape))
    print("[Mag tvec: " + str(format(np.linalg.norm(tvec), ".5f")) + "]")
    print("[Rot_mat: " + str(rot_mat) + " " + str(rot_mat.shape))
    # print("det rot_mat: " + str(np.linalg.det(rot_mat)))
    #
    # euler_angles = rotation_matrix_to_euler_angles(rot_mat, 1e-2)
    #
    # print("euler angles rad: " +
    #       str(format(euler_angles[0], ".5f")) + ":" +
    #       str(format(euler_angles[1], ".5f")) + ":" +
    #       str(format(euler_angles[2], ".5f")))
    # print("euler angles deg: " +
    #       str(format(rad_to_deg(euler_angles[0]), ".3f")) + ":" +
    #       str(format(rad_to_deg(euler_angles[1]), ".3f")) + ":" +
    #       str(format(rad_to_deg(euler_angles[2]), ".3f")))

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


#####################################################################################################################

def main():

    root_dir = 'calib_data'
    extri_ids = ['extri.yml', 'R', 'Rot', 'T']
    intri_ids = ['intri.yml', 'dim', 'K', 'dist']
    calib_image_name = "calib_image.jpg"

    marker_length = 0.2  # in meters
    marker_num = 4

    cv_file = cv_file_io(root_dir)
    cv_file.print_path()
    dir_list = cv_file.get_subdirs(True, True)

    camera_count = cv_file.get_subdir_count()
    print("Camera count: " + str(camera_count))

    extri_easymocap_list = []
    intri_easymocap_list = []

    for i in range(camera_count):

        image_path = dir_list[i] + "/" + calib_image_name

        dim, intrinsic_coeffs, dist_coeffs = cv_file.read_cv_intrinsics_from_yml(dir_list[i] + '/' + intri_ids[0],
                                                                                 intri_ids[1],
                                                                                 intri_ids[2],
                                                                                 intri_ids[3], True)

        tvec, rvec, rot_mat = calc_extrinsics_from_markers(image_path,
                                                           intrinsic_coeffs,
                                                           dist_coeffs,
                                                           marker_length,
                                                           marker_num)

        #output_extrinsics(tvec, rvec, rot_mat, intrinsic_coeffs, dist_coeffs, image_path, i)

        extrinsics_file = dir_list[i] + '/' + extri_ids[0]

        extri_easymocap_list.append([rvec, rot_mat, tvec])
        intri_easymocap_list.append([dim, intrinsic_coeffs, dist_coeffs])

        cv_file.write_cv_extrinsics(extrinsics_file, [extri_ids[1], extri_ids[2], extri_ids[3]], rvec, rot_mat, tvec )

        cv_file.read_cv_extrinsics_from_yml(dir_list[0] + '/' + extri_ids[0],
                                        extri_ids[1],
                                        extri_ids[2],
                                        extri_ids[3], True)

    cv_file.write_cv_intri_easymocap(root_dir, intri_easymocap_list, intri_ids)
    cv_file.write_cv_extri_easymocap(root_dir, extri_easymocap_list, extri_ids)



if __name__== "__main__":
    main()