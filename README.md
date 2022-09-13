# CS 4476/6476 Project 2: SIFT Local Feature Matching

## Getting started

- See [Project 0](https://github.gatech.edu/cs4476/project-0) for detailed environment setup.
- Ensure that you are using the environment `cv_proj2`, which you can install using the install script `conda/install.sh`.

## Logistics

- Submit via [Gradescope](https://gradescope.com).
- Part 5 of this project is **required** for 6476, and **optional** for 4476.
- Additional information can be found in `docs/project-2.pdf`.

## 4476 Rubric

- +20 pts: `part1_harris_corner.py`
- +10 pts: `part2_patch_descriptor.py`
- +10 pts: `part3_feature_matching.py`
- +40 pts: `part4_sift_descriptor.py`
- +20 pts: Report
- -5*n pts: Lose 5 points for every time you do not follow the instructions for the hand-in format


## 6476 Rubric

- +15 pts: `part1_harris_corner.py`
- +10 pts: `part2_patch_descriptor.py`
- +10 pts: `part3_feature_matching.py`
- +40 pts: `part4_sift_descriptor.py`
- +25 pts: Report
- -5*n pts: Lose 5 points for every time you do not follow the instructions for the hand-in format

## Submission format

This is very important as you will lose 5 points for every time you do not follow the instructions.

1. Generate the zip folder (`<your_gt_username>.zip`) for the code portion of your submission once you've finished the project using `python zip_submission.py --gt_username <your_gt_username>`. It should contain:
    - `src/`: directory containing all your code for this assignment
    - `setup.cfg`: setup file for environment, do not need to change this file
    - `additional_data/`: (required for 6476, optional for 4476) the images you took for Part 5, and/or if you use any data other than the images we provide, please include them here
    - `README.txt`: (optional) if you implement any new functions other than the ones we define in the skeleton code (e.g., any extra credit implementations), please describe what you did and how we can run the code. We will not award any extra credit if we can't run your code and verify the results.
2. `<your_gt_username>_proj2.pdf` - your report


## Important Notes

- Please follow the environment setup in [Project 0](https://github.gatech.edu/cs4476/project-0).
- Do **not** use absolute paths in your code or your code will break.
- Use relative paths like the starter code already does.
- Failure to follow any of these instructions will lead to point deductions. Create the zip file by clicking and hand it in with your report PDF through Gradescope (please remember to mark which parts of your report correspond to each part of the rubric).
