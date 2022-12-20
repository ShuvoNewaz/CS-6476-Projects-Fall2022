import argparse
import os
import shutil
import yaml

MEGABYTE = 1024 * 1024

MAX_ARCHIVE_SIZE = 100  # in MB


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
    dir_list = yaml.load(open(".final_zip_dir_list.yml"), Loader=yaml.BaseLoader)
    for dir_name in dir_list["required_directories"]:
        copy(dir_name, "/".join(["temp_submission", dir_name]), True, True)
    for dir_name in dir_list["required_files"]:
        shutil.copyfile(dir_name, "/".join(["temp_submission", dir_name]))
    for dir_name in dir_list["optional_directories"]:
        copy(dir_name, "/".join(["temp_submission", dir_name]), True, False)

    out_file = args.gt_username + ".zip"
    shutil.make_archive(args.gt_username, "zip", "temp_submission")
    if os.path.getsize(out_file) > MAX_ARCHIVE_SIZE * MEGABYTE:
        os.remove(out_file)
        print(f"SUBMISSION DID NOT ZIP, ZIPPED SIZE > {MAX_ARCHIVE_SIZE}MB")
    shutil.rmtree("temp_submission", ignore_errors=True)
