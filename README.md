# AI in Trading NanoDegree (AITND)
This repository contains code for Udacity's [AI in Trading NanoDegree](https://udacity.com/course/ai-for-trading--nd880).

## Repository File Structure
    .
    ├── data/                # Data needs to extracted from the project workspaces in the classroom
    ├── project/             # Code for projects in the classroom
    ├── quiz/                # Code for quizzes in the classroom
    ├── helper.py            # A helper file shared across projects and quizzes
    ├── requierments.txt     # Common packages used for projects and quizzes
    └── tests.py             # Common test functions for unit testing student code in projects and quizzes

## No Data
We don't have a licence to redistribute the data to you. We're working on alternatives to this problem.

## My Repo
This repo is a fork of the udacity repo that complements the AI Trading Nanodegree.

### Braches
The master branch can be cloned and provides enhancements to allow for a clean windows install and instructions for obtaining required input data.
The solutions branch contains all solved projects.

### Installation

For windows:
```
git clone
conda env install requirements\aitnd_windows.yml
python -m ipykernel install --user --name aitnd
```
The last line allows you to run jupyter notebooks using the just installed aitnd python kernel. To check, run
```python
import sys
sys.executable
```
from the command line, as well as a jupyter notebook. Both should return somthing like ```'...\\Anaconda3\\envs\\aitnd\\python.exe'```. You might have to change the kernel used by each notebook manually to aitnd (Kernel -> Change kernel).

### Data

To obtain the input files for project 1-3, open each project's notebook and run
```python
import numpy as np
df = pd.read_csv('../../data/project_1/eod-quotemedia.csv')
df.to_csv('eod-quotemedia.csv')
```
Now you can click on the jupyter icon and download the just created file. Create the `data/project_x` folders in the top directory of your local repo and place the files there. 
For project 4 this is a bit more involved.