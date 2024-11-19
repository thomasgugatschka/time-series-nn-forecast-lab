# To-Do List

A checklist of tasks to accomplish for the project.

- [ ] Add to ReadMe or new *md file which is linked to the README how to add new models -> all relevant changes including model_factory... 

- [ ] **Move Implementation to New Subdirectory**
  - Organize project files for improved structure and readability.

- [ ] Cleanup for Plot_Training -> pngs so they are not merged together from different training runs

- [ ] detailed naming convention for trained models (ts-futur/past/ ...)

- [ ] GPU enabled Training -> if OPTION

- single config file type -> JSON or YAML

-[ ] add pandas. Describe for data check (statistics)

- [ ] add masterthesis.support_files - to this project, if it makes sense..

- [ ] Add ETT Dataset -> cite paper
  - https://github.com/zhouhaoyi/ETDataset
  - also create plots like autocorreclation between the variables
  - preprocessing check script -> and fix?

- [ ] Change paths to be base everything off of the root directory

- [ ] Implement model_factory.py

- [ ] Check how to properly set up a data pipeline

- [ ] Check if it makes sense to have clean entry point for scripts ##if __name__ == "__main__":##
  - check what standard is for such scripts
  - so pack the scripts in "def" and a starting point

- [ ] Check what makes sense to make a package out of it --> __init__.py 

- [ ] the scripts/analysis -> write section within README.md (or new .md file which is integrated into the main README.md)
  - In this section make it that the analysis scripts can be started directly from the readme
  - and explain how the commandline arguments work
- [ ] Check if analyis folder is okay or if notebooks (jypiter) should be used instead for data analysis and intial review
  - for inital review jupyter makes maybe more sense

- [ ] check if the __init__.py files are necessary -> or more are needed? they indicate python packages?!

- [ ] **Refactor Code Structure**

- [ ] **Create Functions in New Python Script for Metric Calculation**
  - Refactor existing code and separate metric calculations into a dedicated Python file.

- [ ] **Docker Setup for Local and Cloud Execution**
  - Research and configure Docker to ensure compatibility both locally and in cloud environments (e.g., Google Cloud).
  - **Notes**: Document any specific configurations required for cloud deployment.

- [ ] **Implement Models: RNN, CNN, Transformer**
  - **Recurrent Neural Network (RNN)**: Develop and test for time-series forecasting.
  - **Convolutional Neural Network (CNN)**: Explore feature extraction potential.
  - **Transformer**: Implement and compare with other models for time-series tasks.

- [ ] **.gitkeep for Folder (Results and Archive)**

- [ ] **Add Scripts for Hyperparameter search (grid search?)**
