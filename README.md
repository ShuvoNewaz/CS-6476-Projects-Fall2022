# Project 6: Classifying Point Clouds Using PointNet

## Getting Started
  - See [Project 0](https://github.gatech.edu/cs4476/project-0) for detailed environment setup.
  - Ensure that you are using the environment cv_proj3, which you can install using the install script conda/install.sh.

---

## Logistics
- Submit via [Gradescope](https://gradescope.com).
- Additional information can be found in `docs/project-6.pdf`.

## Important Notes
- Please follow the environment setup in [Project 0](https://github.gatech.edu/cs4476/project-0).
- Do **not** use absolute paths in your code or your code will break.
- Use relative paths like the starter code already does.
- Failure to follow any of these instructions will lead to point deductions. Create the zip file by clicking and hand it in with your report PDF through Gradescope (please remember to mark which parts of your report correspond to each part of the rubric).

## 4476 Rubric

- Note: Both 4476 and 6476 have the same rubric for this project!
- +15 pts: data loader
- +15 pts: voxel-based baseline model
- +25 pts: simplified PointNet
- +15 pts: analysis
- +15 pts: PointNet with positional encoding
- +15 pts: project 6 report
- -5*n pts: Lose 5 points for every time you do not follow the instructions for the hand-in format

## Submission format

This is very important as you will lose 5 points for every time you do not follow the instructions.

1. Generate the zip folder (`<your_gt_username>.zip`) for the code portion of your submission once you've finished the project using `python zip_submission.py --gt_username <your_gt_username>`. It should contain:
    - `src/`: directory containing all your code for this assignment
    - `setup.cfg`: setup file for environment, do not need to change this file
    - `additional_data/`: (required for 6476, optional for 4476) the images you used for Part 6, and/or if you use any data other than the images we provide, please include them here
2. `<your_gt_username>_proj6.pdf` - your report

## FAQ

### I'm getting [*insert error*] and I don't know how to fix it.

Please check [StackOverflow](https://stackoverflow.com/) and [Google](https://google.com/) first before asking the teaching staff about OS specific installation issues (because that's likely what we will be doing if asked).
