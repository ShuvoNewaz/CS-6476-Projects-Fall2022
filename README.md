# Project 4: Recognition with Deep Learning

## Setup Overview

In the following setup, you will learn how to:

- **Generate** a template repository
- **Clone** your repository
- **Build** your environment
- **Run** the unit tests
- **Submit** assignment to gradescope

---

## Setup

### Generate a template repository

Read about [Template Repositories](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/creating-a-template-repository).

- Click [here](https://github.gatech.edu/cs4476/project-4/generate) or the *Use this template* button at the top right of the repository.
- Name your repository `project-4` and ensure (these are **required!**):
  - You are the **owner**.
  - The repository is **private**.
  
### Clone your repository

From the VSCode terminal, traverse to the directory where you store your code. Enter the following:

```bash
gh repo clone github.gatech.edu/[YOUR_USERNAME]/project-4
```

### Build your environment

Read about [Devcontainers](https://code.visualstudio.com/docs/remote/containers#_quick-start-open-an-existing-folder-in-a-container).

- From terminal, navigate to the root directory of this project `project-4`.
- In your terminal, type `code .`
- You should see a new VSCode instance launch and a prompt in the bottom right corner with a button that says **Reopen in Container**. Click it.
- Your Docker image should start image and you should be good to go after the build process is over!

**NOTE:** You may need to reload VSCode for the `python` extension to finish installing. Verify that the `python` extension does not need to be reloaded in the *extensions* tab.

### Run the unit tests

Read about [pytest](https://docs.pytest.org).

We will use `pytest` in this class as a testing framework. This will run unit tests on your code to verify correctness. Some of the unit tests will given to you as a courtesy; however, others will be run at submission time. The following command will run all unit tests in the `tests` folder.

```bash
pytest tests
```

### Submitting assignment to gradescope

To create the zip file to upload on gradescope, run
```
python submission.py --gt_username your_username
```

### Start your project

The pdf file will help you start the project, so make sure to read it before you start working on the code.

Have fun coding!

---

## FAQ

### I'm getting [*insert error*] and I don't know how to fix it.

Please check [StackOverflow](https://stackoverflow.com/) and [Google](https://google.com/) first before asking the teaching staff about OS specific installation issues (because that's likely what we will be doing if asked).
