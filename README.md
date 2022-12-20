# Project 5: Semantic Segmentation Deep Learning

## Getting Started
  - See [Project 0](https://github.gatech.edu/cs4476/project-0) for detailed environment setup.
  - Ensure that you are using the environment cv_proj5, which you can install using the install script conda/install.sh.

---

## Logistics
- Submit via [Gradescope](https://gradescope.com).
- Part 6 (transfer learning) of this project is **optional** (extra credit) for 4476 and **required** for 6476.
- Additional information can be found in `docs/project-5.pdf`.

## Important Notes
- Please follow the environment setup in [Project 0](https://github.gatech.edu/cs4476/project-0).
- Do **not** use absolute paths in your code or your code will break.
- Use relative paths like the starter code already does.
- Failure to follow any of these instructions will lead to point deductions. Create the zip file by clicking and hand it in with your report PDF through Gradescope (please remember to mark which parts of your report correspond to each part of the rubric).

## 4476 Rubric

- +10 pts: Part 1 Code
- +5 pts: Part 2 Code
- +10 pts: Part 3 Code
- +15 pts: Part 4 Code
- +40 pts: Part 5 Code (+32 pts: PSPNet model accuracy. you'll need to reach 60\% mIoU on your submitted grayscale PNG 
label maps to get full credit. You will get partial credit if your mIoU is lower than the threshold. 
score = mIoU/threshold X 32)
- +20 pts: Report
- -5*n pts: Lose 5 points for every time you do not follow the instructions for the hand-in format


## 6476 Rubric

- +10 pts: Part 1 Code
- +5 pts: Part 2 Code
- +10 pts: Part 3 Code
- +10 pts: Part 4 Code
- +40 pts: Part 5 Code (+32 pts: PSPNet model accuracy. you'll need to reach 60\% mIoU on your submitted grayscale PNG 
label maps to get full credit. You will get partial credit if your mIoU is lower than the threshold. 
score = mIoU/threshold X 32)
- +5 pts: Part 6 Code
- +20 pts: Report
- -5*n pts: Lose 5 points for every time you do not follow the instructions for the hand-in format

## Submission format

This is very important as you will lose 5 points for every time you do not follow the instructions.

1. Generate the zip folder (`<your_gt_username>.zip`) for the code portion of your submission once you've finished the project using `python zip_submission.py --gt_username <your_gt_username>`. It should contain:
    - `src/`: directory containing all your code for this assignment
    - `setup.cfg`: setup file for environment, do not need to change this file
    - `grayscale_predictions.zip`: after running Colab, this zip file will be created with your model's outputs on the Camvid validation set (remember to download before closing your session!)
    - `exp/kitti/PSPNet/model/kitti_result/` - (optional for 4476) directory containing KITTI transfer learning results (remember to download before closing your session!)
    - `additional_data/`: (optional) if you use any data other than the images we provide you, please include them here
    - `README.txt`: (optional) if you implement any new functions other than the ones we define in the skeleton code (e.g., any extra credit implementations), please describe what you did and how we can run the code. We will not award any extra credit if we can't run your code and verify the results.
2. `<your_gt_username>_proj5.pdf` - your report

## FAQ

### I'm getting [*insert error*] and I don't know how to fix it.

Please check [StackOverflow](https://stackoverflow.com/) and [Google](https://google.com/) first before asking the teaching staff about OS specific installation issues (because that's likely what we will be doing if asked).
