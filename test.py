from tools.cv_file_io import file_io


def main():

    root_path = 'calib_data/input'
    intrisincs_file_name = 'extrinsics.yml'
    intrinsics_id_list = ['intrisics.yml', 'T', 'R', 'Rot']

    file = file_io(root_path)

    file.print_path()

    dir_list = file.get_subdirs(root_path, True, True)

    file.read_cv_intrinsics_from_yml(dir_list[0] + '/' + intrisincs_file_name,
                                    intrinsics_id_list[1],
                                    intrinsics_id_list[2],
                                    intrinsics_id_list[3], True)




if __name__== "__main__":
   main()