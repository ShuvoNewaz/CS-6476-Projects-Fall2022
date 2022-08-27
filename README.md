# CS-4476/6476 Project 1: Convolution and Hybrid Images

## Getting Started

- See [Project 0](https://github.gatech.edu/cs4476/project-0) for detailed environment setup. This project's environment is set up similarly, and will create a conda environment called `cv_proj1`.
- Ensure that you are using the environment `cv_proj1`, which you can install using the install script `conda/install.sh`.

## Logistics

- Submit via [Gradescope](https://gradescope.com).
- Part 4 of this project is **required** for 6476, and **optional** for 4476.
- Additional information can be found in `docs/project-1.pdf`.

## 4476 Rubric

- +5 pts: `create_Gaussian_kernel_1D()` in `part1.py`
- +5 pts: `create_Gaussian_kernel_2D()` in `part1.py`
- +15 pts: `my_conv2d_numpy()` in `part1.py`
- +10 pts: `create_hybrid_image()` in `part1.py`
- +5 pts: `make_dataset()` in `part2_datasets.py`
- +5 pts: `get_cutoff_frequencies()` in `part2_datasets.py`
- +5 pts: `__len__()` in `part2_datasets.py`
- +5 pts: `__getitem__()` in `part2_datasets.py`
- +5 pts: `get_kernel()` in `part2_models.py`
- +5 pts: `low_pass()` in `part2_models.py`
- +10 pts: `forward()` in `part2_models.py`
- +5 pts: `my_conv2d_pytorch()` in `part3.py`
- +20 pts: Report
- -5*n pts: Lose 5 points for every time you do not follow the instructions for the hand-in format


## 6476 Rubric

- +4 pts: `create_Gaussian_kernel_1D()` in `part1.py`
- +4 pts: `create_Gaussian_kernel_2D()` in `part1.py`
- +15 pts: `my_conv2d_numpy()` in `part1.py`
- +10 pts: `create_hybrid_image()` in `part1.py`
- +4 pts: `make_dataset()` in `part2_datasets.py`
- +4 pts: `get_cutoff_frequencies()` in `part2_datasets.py`
- +3 pts: `__len__()` in `part2_datasets.py`
- +3 pts: `__getitem__()` in `part2_datasets.py`
- +4 pts: `get_kernel()` in `part2_models.py`
- +4 pts: `low_pass()` in `part2_models.py`
- +10 pts: `forward()` in `part2_models.py`
- +5 pts: `my_conv2d_pytorch()` in `part3.py`
- +5 pts: `my_cond2d_freq()` in `part4.py`
- +5 pts: `my_deconv2d_freq()` in `part4.py`
- +20 pts: Report
- -5*n pts: Lose 5 points for every time you do not follow the instructions for the hand-in format


## Submission format

This is very important as you will lose 5 points for every time you do not follow the instructions.

1. Generate the zip folder (`<your_gt_username>.zip`) for the code portion of your submission once you've finished the project using `python zip_submission.py --gt_username <your_gt_username>`. It should contain:
    - `src/`: directory containing all your code for this assignment
    - `cutoff_frequency.txt`: .txt file containing the best cutoff frequency values you found for each pair of images in `data/`
    - `setup.cfg`: setup file for environment, do not need to change this file
    - `additional_data/`: (optional) if you use any data other than the images we provide, please include them here
    - `README.txt`: (optional) if you implement any new functions other than the ones we define in the skeleton code (e.g., any extra credit implementations), please describe what you did and how we can run the code. We will not award any extra credit if we can't run your code and verify the results.
2. `<your_gt_username>_proj1.pdf` - your report


## Important Notes

- Please follow the environment setup in [Project 0](https://github.gatech.edu/cs4476/project-0).
- Do **not** use absolute paths in your code or your code will break.
- Use relative paths like the starter code already does.
- Failure to follow any of these instructions will lead to point deductions. Create the zip file by clicking and hand it in with your report PDF through Gradescope (please remember to mark which parts of your report correspond to each part of the rubric).

## Project Structure

```console
.
├── README.md
├── cutoff_frequencies.txt
├── data
│   ├── 1a_dog.bmp
│   ├── 1b_cat.bmp
│   ├── 2a_motorcycle.bmp
│   ├── 2b_bicycle.bmp
│   ├── 3a_plane.bmp
│   ├── 3b_bird.bmp
│   ├── 4a_einstein.bmp
│   ├── 4b_marilyn.bmp
│   ├── 5a_submarine.bmp
│   └── 5b_fish.bmp
│   └── part4
│       ├── kernel.npy
│       └── mystery.npy
├── docs
│   └── report.pptx
├── project-1.ipynb
├── pyproject.toml
├── scripts
│   └── submission.py
├── setup.cfg
├── src
│   └── vision
│       ├── __init__.py
│       ├── part1.py
│       ├── part2_datasets.py
│       ├── part2_models.py
│       ├── part3.py
│       └── utils.py
└── tests
    ├── __init__.py
    ├── __pycache__
    ├── test_part1.py
    ├── test_part2.py
    ├── test_part3.py
    └── test_utils.py
```
