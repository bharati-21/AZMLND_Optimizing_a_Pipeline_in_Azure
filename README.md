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
      1. [#### Hyperparameter Tuning using HyperDrive](#hperparameter-tuning-using-hyperDrive)
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

### Scikit-Learn Principle:
#### Data
- First, the dataset was loaded from the given [URL](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) into the notebook using the _TabularDatasetFactory_ class
- The given dataset was then cleaned using the clean_data() method predefined in the [train.py file](https://github.com/bharati-21/AZMLND_Optimizing_a_Pipeline_in_Azure/blob/master/train.py) that performed various preprocessing steps (such as one hot encoding) on the data, after which the data wass split into train and test sets in 80-20 ratio

#### Scikit-Learn Logistic Regression
- The cleaned train data was then fed to the scikit-learn based logistic regression algorithm which took 2 hyperparameters:
  1. **--C:** This parameter specifies inverse regularization strength that helps in addressing the overfitting problem which commonly occurs in Machine Learning. Regularization helps in avoiding overfitting by shrinking the model's coefficients towards zero. A larger C implies less regularization; while a smaller C means more regularization. 
	1. **--max_iter:** This parameter states the maximum number of iterations that should be taken to converge

#### Hyperparameter Tuning using HyperDrive
- To select the best hyperparameters for the model, the HyperDrive was used which automated parameter tuning
- The steps used for effective hyperparameter tuning using the HyperDrive package were:
1. **Defining the parameter search space:**
   - This is done by exploring the range of values defined for each hyperparameter
   - Range of Hyperparameters depends on the type of the parameter- whether it's discrete or continuous
   - _This project uses the `discrete search space for --C` parameter and a `continuous search space for --max_iter_`
1. **Sampling the Hyperparameter space:**
   - To find these values over the specified search space a sampling method is used. Azure ML supports `Random sampling`, `Grid sampling`, and `Bayesian sampling`
   - _This project uses the Random Sampling method to choose values from the mentioned search space_
1. **Specifying a primary metric:**
   - A primary metric is specified to optimize hyperparameter tuning. 
   - Each training run is evaluated for the primary metric. 
   - The attributes specified are: 
     - primary_metric_name: Name of the primary metric needs to exactly match the name of the metric logged by the training script
     - primary_metric_goal: It can be either `PrimaryMetricGoal.MAXIMIZE` or `PrimaryMetricGoal.MINIMIZE` (to maximize or minimize the primary metric when evaluating the runs.)
1. **Specifying early termination policy:**
   - Early termination policy forces low performing runs to automatically terminate so that resources are not wasted on runs that will not give potential results
   - _This project uses the `Bandit Policy` which is based on a slack factor/slack amount and an evaluation interval. Bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run._
1. **Allocating resources**
   - Resources are allocated and controlled for running the hyperparameter tuning experiment
   - The resources to be specified are: 
     - max_total_runs: Maximum number of training runs. (An integer between 1 and 1000)
     - max_duration_minutes: Maximum duration, in minutes, of the hyperparameter tuning experiment. Runs yet to be finished after this duration are canceled
1. **Configuring the experiment**
   - The attributes required to configure the hyperparameter tuning experiment are: `hyperparameter search space`, `Early termination policy`, `Primary metric`, `Resource allocation settings`, `Estimator` 
  - This project uses the SKLearn() method as an estimator that begins the training and invokes the script file.
1. **Submitting the experiment**
  - The experiment is submitted via the `Experiment` class which acts as the entry point for creating and working with experiments. 
1. **Visualizing the training runs**
   - The Notebook widget is used to visualize the experiment, run progress, logs and model metrics.
1. **Selecting the configurations of the best model**
   - Once all of the hyperparameter tuning runs completes, the best performing configuration and hyperparameter values are identified

- **What are the benefits of the parameter sampler you chose?**
- **What are the benefits of the early stopping policy you chose?**

### AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline Comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future Work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of Cluster Clean Up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
