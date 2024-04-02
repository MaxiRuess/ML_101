# ML_101
 
This Repo is designed to be a guide for different Machine Learning Models and Algorithms. 
All Models have a 
    - Notebook - Jupyter Notebook with Step-by-Step Instructions
    - Script - Python files to show how to deploy models 


The Repo starts of very simple with LinearRegression and progressivly gets more advanced. 
I'm trying explain the Algorithms as best as possible. Many of the text here have been written with help from ChatGPT. 

You can use the code with the supplied datasets but I've also tried to make the code modular and unviersal applicable to other datasets. (There are some limitations with this especially with Preprocessing the data). The only place where changes are required are in the YAML Config file. I've set it up this way so you can limit the specific what columns you want to include as features

This Repo is a work in progress and I will continously update it with more Notebooks and Scripts

## Instructions 

1. install `requirements.txt`
2. If you using the supplied datasets you can run the scripts from the terminal and the Notebooks directly 
3. If you're using other datasets, you need to adjust the Features and Target variable in the YAML file. 
4. In the `config.yaml` file you can also adjust the hyperparamters 

##Current Models: 

### Regression 

- LinearRegression (OLS)
- Gradient Boosting Regressor (XGBoost)
- Random Forest Regressor (Build with Streamlit as an endpoint)

### Classification 

- Random Forest Classifier
- Gradient Boosting Classifier (XGBoost)


### Reinforcement Learning

- Q-Learning
