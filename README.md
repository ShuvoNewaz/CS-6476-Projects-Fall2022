# Project 3: Camera Calibration and Fundamental Matrix Estimation with RANSAC

## Getting Started
  - See [Project 0](https://github.gatech.edu/cs4476/project-0) for detailed environment setup.
  - Ensure that you are using the environment cv_proj3, which you can install using the install script conda/install.sh.

---

## Logistics
- Submit via [Gradescope](https://gradescope.com).
- Part 6 (panorama stitching) of this project is **optional** (extra credit) for 4476 and **required** for 6476.
- Additional information can be found in `docs/project-3.pdf`.

## Important Notes
- Please follow the environment setup in [Project 0](https://github.gatech.edu/cs4476/project-0).
- Do **not** use absolute paths in your code or your code will break.
- Use relative paths like the starter code already does.
- Failure to follow any of these instructions will lead to point deductions. Create the zip file by clicking and hand it in with your report PDF through Gradescope (please remember to mark which parts of your report correspond to each part of the rubric).

## 4476 Rubric

- +20 pts: `part1_projection_matrix.py`
- +25 pts: `part2_fundamental_matrix.py`
- +30 pts: `part3_ransac.py`
- +25 pts: Report
- -5*n pts: Lose 5 points for every time you do not follow the instructions for the hand-in format


## 6476 Rubric

- +20 pts: `part1_projection_matrix.py`
- +25 pts: `part2_fundamental_matrix.py`
- +25 pts: `part3_ransac.py`
- +5 pts: `part6_panorama_stitching.py`
- +25 pts: Report
- -5*n pts: Lose 5 points for every time you do not follow the instructions for the hand-in format

## Submission format

This is very important as you will lose 5 points for every time you do not follow the instructions.

1. Generate the zip folder (`<your_gt_username>.zip`) for the code portion of your submission once you've finished the project using `python zip_submission.py --gt_username <your_gt_username>`. It should contain:
    - `src/`: directory containing all your code for this assignment
    - `setup.cfg`: setup file for environment, do not need to change this file
    - `additional_data/`: (required for 6476, optional for 4476) the images you used for Part 6, and/or if you use any data other than the images we provide, please include them here
2. `<your_gt_username>_proj3.pdf` - your report

## FAQ

### I'm getting [*insert error*] and I don't know how to fix it.

Please check [StackOverflow](https://stackoverflow.com/) and [Google](https://google.com/) first before asking the teaching staff about OS specific installation issues (because that's likely what we will be doing if asked).
