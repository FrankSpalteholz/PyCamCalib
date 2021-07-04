import cv2
import os

class cv_file_io:

    def __init__(self, data_path):
        self.data_path = data_path
        self.file_list = []
        self.yaml_header = "%YAML:1.0\n---\n"
        self.cv_matrix_header = ": !!opencv-matrix\n"
        self.cam_names_id = "names:\n"
        self.cv_matrix_ids = ["rows: ", "cols: ", "dt: ", "data: "]

    def print_path(self):
        print("[Data root path: " + self.data_path + "]")

    def get_subdirs(self, sorted, is_debug):
        for file in os.listdir(self.data_path):
            d = os.path.join(self.data_path, file)
            if os.path.isdir(d):
                self.file_list.append(d)
        if sorted:
            self.file_list.sort(key=lambda x: int(x.rsplit('.', 1)[1]))
        if is_debug:
            print("========================================================================")
            print("[SubDirs: " + self.data_path + "]\n")
            for i in range(len(self.file_list)):
                print("[SubDir: " + str(i) + "] " + self.file_list[i])
        return self.file_list

    def get_subdir_count(self):
        return len(self.file_list)

    def read_cv_extrinsics_from_yml(self, file_path, rvec_name, rot_mat_name, tvec_name, is_debug):
        fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
        print(file_path)
        fn_r = fs.getNode(rvec_name).mat()
        fn_rot = fs.getNode(rot_mat_name).mat()
        fn_t = fs.getNode(tvec_name).mat()

        if is_debug:
            print("========================================================================")
            print("[Extrinsics: " + file_path + "]\n")
            print("[Rvec: " + str(fn_r))
            print("[Tvec: " + str(fn_t))
            print("[RotMat: " + str(fn_rot))

        return fn_t, fn_r, fn_rot

    def read_cv_intrinsics_from_yml(self, file_path, dimensions_name, camera_matrix_name, dist_coeffs_name, is_debug):
        fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
        fn_dim = fs.getNode(dimensions_name).mat()
        fn_camera_mat = fs.getNode(camera_matrix_name).mat()
        fn_dist_coeffs = fs.getNode(dist_coeffs_name).mat()

        if is_debug:
            print("========================================================================")
            print("[Intrinsics: " + file_path + "]\n")
            print("[Image width/ height: " + str(fn_dim))
            print("[K Coeffs: " + str(fn_camera_mat))
            print("[Dist Coeffs: " + str(fn_dist_coeffs))

        return fn_dim, fn_camera_mat, fn_dist_coeffs

    def write_cv_extrinsics(self, file_path, id_names, rvec, rot_mat, tvec):
        file1 = open(file_path, "w")
        file1.write(self.yaml_header)

        # write rvec
        file1.write(id_names[0] + self.cv_matrix_header)
        file1.write("    " + self.cv_matrix_ids[0] + "3\n")
        file1.write("    " + self.cv_matrix_ids[1] + "1\n")
        file1.write("    " + self.cv_matrix_ids[2] + "d\n")
        file1.write("    " + self.cv_matrix_ids[3] + "[ " +
                    str(rvec[0]).replace("[","").replace("]","") + ", " +
                    str(rvec[1]).replace("[","").replace("]","") + ", " + "\n" +
                    "        " + str(rvec[2]).replace("[","").replace("]","") + " ]\n")

        # write rot_mat
        file1.write(id_names[1] + self.cv_matrix_header)
        file1.write("    " + self.cv_matrix_ids[0] + "3\n")
        file1.write("    " + self.cv_matrix_ids[1] + "3\n")
        file1.write("    " + self.cv_matrix_ids[2] + "d\n")
        file1.write("    " + self.cv_matrix_ids[3] + "[ " +
                    str(rot_mat[0][0]) + ", " + str(rot_mat[0][1]) + ", " + "\n" +
                    "        " + str(rot_mat[0][2]) + ", " + str(rot_mat[1][0]) + ", " + "\n" +
                    "        " +str(rot_mat[1][1]) + ", " + str(rot_mat[1][2]) + ", " + "\n" +
                    "        " +str(rot_mat[2][0]) + ", " + str(rot_mat[2][1]) + ", " + "\n" +
                    "        " +str(rot_mat[2][2]) + " ]\n")

        # write tvec
        file1.write(id_names[2] + self.cv_matrix_header)
        file1.write("    " + self.cv_matrix_ids[0] + "3\n")
        file1.write("    " + self.cv_matrix_ids[1] + "1\n")
        file1.write("    " + self.cv_matrix_ids[2] + "d\n")
        file1.write("    " + self.cv_matrix_ids[3] + "[ " +
                    str(tvec[0][0][0]) + ", " +
                    str(tvec[0][0][1]) + ", " + "\n" +
                    "        " + str(tvec[0][0][2]) + " ]\n")

        file1.close()

    def write_cv_intri_easymocap(self, root_dir, intri_list, intri_ids):
        file1 = open(root_dir + "/" + intri_ids[0], "w")
        file1.write(self.yaml_header)
        file1.write(self.cam_names_id)
        for i in range(len(intri_list)):
            file1.write("   - " + "\"" + str(i+1) + "\"\n")

        for i in range(len(intri_list)):
            # write K
            file1.write(intri_ids[2] + "_" + str(i+1) + self.cv_matrix_header)
            file1.write("    " + self.cv_matrix_ids[0] + "3\n")
            file1.write("    " + self.cv_matrix_ids[1] + "3\n")
            file1.write("    " + self.cv_matrix_ids[2] + "d\n")
            file1.write("    " + self.cv_matrix_ids[3] + "[ " +
                        str(intri_list[i][1][0][0]) + ", " + str(intri_list[i][1][0][1]) + ", " + "\n" +
                        "        " + str(intri_list[i][1][0][2]) + ", " + str(intri_list[i][1][1][0]) + ", " + "\n" +
                        "        " + str(intri_list[i][1][1][1]) + ", " + str(intri_list[i][1][1][2]) + ", " + "\n" +
                        "        " + str(intri_list[i][1][2][0]) + ", " + str(intri_list[i][1][2][1]) + ", " + "\n" +
                        "        " + str(intri_list[i][1][2][2]) + " ]\n")

            # write dist
            file1.write(intri_ids[3] + "_" + str(i+1) + self.cv_matrix_header)
            file1.write("    " + self.cv_matrix_ids[0] + "5\n")
            file1.write("    " + self.cv_matrix_ids[1] + "1\n")
            file1.write("    " + self.cv_matrix_ids[2] + "d\n")
            file1.write("    " + self.cv_matrix_ids[3] + "[ " +
                        str(intri_list[i][2][0]).replace("[", "").replace("]", "") + ", " +
                        str(intri_list[i][2][1]).replace("[", "").replace("]", "") + ", " + "\n" +
                        "        " + str(intri_list[i][2][2]).replace("[", "").replace("]", "") + ", " +
                        str(intri_list[i][2][3]).replace("[", "").replace("]", "") + ", " + "\n" +
                        "        " + str(intri_list[i][2][4]).replace("[", "").replace("]", "") + " ]\n")

        file1.close()

    def write_cv_extri_easymocap(self, root_dir, extri_list, extri_ids):
        file1 = open(root_dir + "/" + extri_ids[0], "w")
        file1.write(self.yaml_header)
        file1.write(self.cam_names_id)
        for i in range(len(extri_list)):
            file1.write("   - " + "\"" + str(i + 1) + "\"\n")

        for i in range(len(extri_list)):
            # write rvec
            file1.write(extri_ids[1] + "_" + str(i+1) + self.cv_matrix_header)
            file1.write("    " + self.cv_matrix_ids[0] + "3\n")
            file1.write("    " + self.cv_matrix_ids[1] + "1\n")
            file1.write("    " + self.cv_matrix_ids[2] + "d\n")
            file1.write("    " + self.cv_matrix_ids[3] + "[ " +
                        str(extri_list[i][0][0]).replace("[", "").replace("]", "") + ", " +
                        str(extri_list[i][0][1]).replace("[", "").replace("]", "") + ", " + "\n" +
                        "        " + str(extri_list[i][0][2]).replace("[", "").replace("]", "") + " ]\n")

            # write rot_mat
            file1.write(extri_ids[2] + "_" + str(i+1) + self.cv_matrix_header)
            file1.write("    " + self.cv_matrix_ids[0] + "3\n")
            file1.write("    " + self.cv_matrix_ids[1] + "3\n")
            file1.write("    " + self.cv_matrix_ids[2] + "d\n")
            file1.write("    " + self.cv_matrix_ids[3] + "[ " +
                        str(extri_list[i][1][0][0]) + ", " + str(extri_list[i][1][0][1]) + ", " + "\n" +
                        "        " + str(extri_list[i][1][0][2]) + ", " + str(extri_list[i][1][1][0]) + ", " + "\n" +
                        "        " + str(extri_list[i][1][1][1]) + ", " + str(extri_list[i][1][1][2]) + ", " + "\n" +
                        "        " + str(extri_list[i][1][2][0]) + ", " + str(extri_list[i][1][2][1]) + ", " + "\n" +
                        "        " + str(extri_list[i][1][2][2]) + " ]\n")

            # write tvec
            file1.write(extri_ids[3] + "_" + str(i+1) + self.cv_matrix_header)
            file1.write("    " + self.cv_matrix_ids[0] + "3\n")
            file1.write("    " + self.cv_matrix_ids[1] + "1\n")
            file1.write("    " + self.cv_matrix_ids[2] + "d\n")
            file1.write("    " + self.cv_matrix_ids[3] + "[ " +
                        str(extri_list[i][2][0][0][0]) + ", " +
                        str(extri_list[i][2][0][0][1]) + ", " + "\n" +
                        "        " + str(extri_list[i][2][0][0][2]) + " ]\n")

        file1.close()