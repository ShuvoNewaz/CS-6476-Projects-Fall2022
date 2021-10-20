import argparse
import os
import shutil
import yaml


def copy(src: str, dest: str, directory: bool, required: bool):
    ds = "Directory" if directory else "File"

    def handle_error(e):
        if required:
            print("%s not copied. Error: %s" % (ds, e))
            print("Exiting, SUBMISSION NOT ZIPPED!")
            shutil.rmtree("temp_submission", ignore_errors=True)
            exit()
        else:
            print("%s %s missing but optional, skipping." % (ds, src))

    try:
        if directory:
            shutil.copytree(src, dest)
        else:
            shutil.copy(src, dest)
    except shutil.Error as e:
        handle_error(e)
    except OSError as e:
        handle_error(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gt_username", required=True, type=str)

    args = parser.parse_args()

    shutil.rmtree("temp_submission", ignore_errors=True)
    os.mkdir("temp_submission")
    dir_list = yaml.load(open(".zip_dir_list.yml"), Loader=yaml.BaseLoader)
    for dir_name in dir_list["required_directories"]:
        copy(dir_name, "/".join(["temp_submission", dir_name]), True, True)
    for file_name in dir_list["required_files"]:
        copy(file_name, "/".join(["temp_submission", file_name]), False, True)
    for dir_name in dir_list["optional_directories"]:
        copy(dir_name, "/".join(["temp_submission", dir_name]), True, False)
    for file_name in dir_list["optional_files"]:
        copy(file_name, "/".join(["temp_submission", file_name]), False, False)

    out_file = args.gt_username + ".zip"
    shutil.make_archive(args.gt_username, "zip", "temp_submission")
    if os.path.getsize(out_file) > 60000000:
        os.remove(out_file)
        print("SUBMISSION DID NOT ZIP, ZIPPED SIZE > 15MB")
    shutil.rmtree("temp_submission", ignore_errors=True)
