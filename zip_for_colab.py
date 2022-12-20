import os
import shutil
import yaml

FILE_SIZE_LIMIT = 400000000
OUT_FILENAME = "cv_proj5_colab"
FILE_LIST_YAML = ".colab_zip_dir_list.yml"


def copy(src: str, dst: str, directory: bool, required: bool) -> None:
    """ Copies contents from source to destination directory"""
    ds = "Directory" if directory else "File"

    def handle_error(e):
        if required:
            print("%s not copied. Error: %s" % (ds, e))
            print("Exiting, COLAB UPLOAD NOT ZIPPED!")
            shutil.rmtree("temp_colab_upload", ignore_errors=True)
            exit()
        else:
            print("%s %s missing but optional, skipping." % (ds, src))

    try:
        if directory:
            shutil.copytree(src, dst)
        else:
            shutil.copy(src, dst)
    except shutil.Error as e:
        handle_error(e)
    except OSError as e:
        handle_error(e)


def main() -> None:
    shutil.rmtree("temp_colab_upload", ignore_errors=True)
    os.mkdir("temp_colab_upload")
    dir_list = yaml.load(open(FILE_LIST_YAML), Loader=yaml.BaseLoader)
    
    for dir_name in dir_list["required_directories"]:
        if (dir_name == "src/vision"):
            copy(src=dir_name, dst="/".join(["temp_colab_upload", "vision"]), directory=True, required=True)
        else:
            copy(src=dir_name, dst="/".join(["temp_colab_upload", dir_name]), directory=True, required=True)

    for fpath in dir_list["required_files"]:
        copy(src=fpath, dst="/".join(["temp_colab_upload", fpath]), directory=False, required=True)


    out_file = f"{OUT_FILENAME}.zip"
    shutil.make_archive(OUT_FILENAME, "zip", "temp_colab_upload")
    if os.path.getsize(out_file) > FILE_SIZE_LIMIT:
        os.remove(out_file)
        print(f"COLAB_UPLOAD DID NOT ZIP, ZIPPED SIZE > {(FILE_SIZE_LIMIT//1e6)}MB")
    shutil.rmtree("temp_colab_upload", ignore_errors=True)


if __name__ == "__main__":
    main()
