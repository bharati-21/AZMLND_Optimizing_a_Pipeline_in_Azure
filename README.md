# OPTIMIZING AN ML PIPELINE IN AZURE

## TABLE OF CONTENTS
1. [Overview](#overview)
1. [Summary](#summary)
   1. [Problem](#problem)
   1. [Solution Summary](#solution-summary)
   1. [Result Summary](#result-summary)
1. [Approaches](#approaches)
   1. [Scikit-Learn Pipeline](#scikit-learn-pipeline)
      1. [Data](#data)
      1. [Scikit-Learn Logistic Regression](#scikit-learn-logistic-regression)
      1. [Hyperparameter Tuning using HyperDrive](#hperparameter-tuning-using-hyperDrive)
1. [AutoML](#automl)
1. [Pipeline Comparison](#pipeline-comparison)
1. [Future Work](#future-work)
1. [Proof of Cluster Clean Up](#proof-of-cluster-clean-up)


## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
### Problem
- This project uses a Bank Marketing Dataset from the [USI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). 
- The dataset conatins personal details about their clients such as age, job, marital status, education, etc among other attributes. 
- This is a **_classification_** (2 class-classification) problem with the goal to predict whether or not a client will subscribe to a term deposit with the bank. 
- The data is classified using the column label y in the dataset that contains binary valuse ('yes' and 'no')**

### Solution Summary
- This project used two approaches to find the best possible model for classifying the given dataset:
  - Scikit-Learn based logistic regression which used the HyperDrive for effective hyperparameter tuning
  - Automated Machine Learning was used to build and choose the best model
 
### Result Summary
* The best performing model was a VotingEnsemble algorithm that was selected through AutoML with an accuracy of 0.91524
* The Logistic Regression model gave an accuracy of 0.91442 and the hyperparameters for the model were selectrd using HyperDrive

## Approaches
- Two approaches were used in this project to classify the given data:
  1. A Scikit-Learn based Logistic Regression algorithm was used to classify the data and the hyperparameters of the model were tuned using the HyperDrive
  1. Automated ML was used to come up with the best possible algorithm and hyperparameters to find the best model
- Both these approaches were extecuted using Jupyter Notebook and the Azure ML SDK.

### Scikit-Learn Pipeline:

#### Data Preparation
- First, the dataset is loaded from the given [URL](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) into the notebook using the _TabularDatasetFactory_ class
- The given dataset is then cleaned using the clean_data() method predefined in the [train.py file](https://github.com/bharati-21/AZMLND_Optimizing_a_Pipeline_in_Azure/blob/master/train.py) that performs various preprocessing steps (such as one hot encoding) on the data, after which the data is split into train and test sets in 70-30 ratio

#### Scikit-Learn Logistic Regression Algorithm
- The split train data is then fed to the scikit-learn based logistic regression algorithm which takes in 2 hyperparameters: `--C`, which is the inverse of regularization strength and `--max-iter`, which is the maximum number of iterations that should be taken to converge.

#### Hyperparameter Tuning using HyperDrive
- The HyperDrive package is used to optimize hyperparameter tuning by using the HyperDriveConfig() that takes in several configuration attributes:
  1. Estimator (`est`): An `SKLearn` estimator is used to begin the training and invoke the training script file.
  1. Parameter sampler (`hyperparameter_sampling `): A `RandomParameterSampling` sampler is used to randomly select values specified in the search space for the two parameters of Logistic Regression algorithm (--c and --max_iter)
  1. Policy (`policy`): An early termination policy, `BanditPolicy`, is passed to ensure low performing runs are terminated and resources are not wasted
  1. Primary Metric (`primary_metric_name`): The primary metric for evaluating runs is specified. The project uses `accuracy` as the primary metric with the goal (`primary_metric_goal`) value `primary_metric_goal.MAXIMIZE` to maximize the primary metric in every run
  1. Resources for controlling and running the experiment is specified using `max_concurrent_runs` (Maximum number of runs that can run concurrently in the experiment) and `max_total_runs` (Maximum number of training runs). 

#### Submitting and Saving the best model
- The Hyperdrive run is then submitted to the experiment which takes the hyperdrive configuration details as the parameter. Once the run is completed, the best metrics are obtained using `run.get_best_run_by_primary_metric()` and the model is tested for primary_metric (accuracy) using the test data from the script file. The best run is then registered after invoking `register_model()`.

- **What are the benefits of the parameter sampler you chose?**
  - Random Sampling works with both discrete and continous search space unlike Grid Sampling. It also supports early termination policy unlike Bayesian Sampling. Hence Random Sampler helps in performing trial and error with values chosen over the search space and then refine the search space to obtain best results.
- **What are the benefits of the early stopping policy you chose?**
  - Early stopping helps in avoiding unnecessary usage of resources assigned to runs that performs poorly. This is ensured by terminating runs whose primary metric is not within the slack amount specified by the policy. 

### AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**
- AutoML (Automated Machine Learning) is used to simplify various time intensive and inexhaustive Machine Learning tasks such as feature engineering, feature selection, hyperparameter selection, training, testing etc. AutoML helps in training over hundereds of models in a single day rather than manually training and waiting for obtaining an accurate model.  
- The same Bank Marketing Dataset from the [USI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) is classified again by using AutoML.  
- AutoML is used to automate the process of choosing an algorithm and the values for the chosen algorithm's hyperparameters, that will result in the best model which classifies the given dataset. 
- The dataset is uploaded from the URL via the _TabularDatasetFactory_ class. The data is then cleaned using clean_data from the [train.py file](https://github.com/bharati-21/AZMLND_Optimizing_a_Pipeline_in_Azure/blob/master/train.py), and then split into train and test sets in 70-30 ratio 
- The `AutoMLConfig` object takes attributes required to configure the experiement run such as: 
  1. Experiment Timeout (`experiment_timeout_minutes`): Maximum amount of time (in minutes) that all iterations combined can take before the experiment terminates. 
  1. Task to be performed (`task`): The tpye of task that needs to be run such as classification, regression, forecasting etc. In this project `classification` is the task to be performed.
  1. Primary Metric (`primary_metric`): The primary metric which is used to evaluate every run. In this case, `accuracy` is the primary metric to be evaluated.
  1. Training Data (`training_data`) = The _TabularDataset_ that contains the training data,
  1. Label Column (`label_column_name`): Name of the column that needs to be predicted. In this case the column that contains "yes" or "no" to perform classification.
  1. Cross Validations (`n_cross_validations`): Specifies the number of cross validations that needs to be performed on each model by splitting the dataset into n subsets.
  1. Compute Target (`compute_target`): The cluster used to run the experiment on. 
- The algorithms that were used to train the model were:
  1. Algorithms used
  1. LightGBM
  1. XGBoostClassifier
  1. RandomForest
  1. LightGBM
  1. LogisticRegression
  1. RandomForest

## Pipeline Comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future Work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of Cluster Clean Up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
