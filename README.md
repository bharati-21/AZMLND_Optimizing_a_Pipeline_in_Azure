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
- First, the dataset was loaded from the given [URL](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) into the notebook using the _TabularDatasetFactory_ class
- The given dataset was then cleaned using the clean_data() method predefined in the [train.py file](https://github.com/bharati-21/AZMLND_Optimizing_a_Pipeline_in_Azure/blob/master/train.py) that performed various preprocessing steps (such as one hot encoding) on the data, after which the data wass split into train and test sets in 80-20 ratio
#### Scikit-Learn Logistic Regression
- The split train data was then fed to the scikit-learn based logistic regression algorithm which took 2 hyperparameters: `--C`, which is the inverse of regularization strength and `--max-iter`, which is the maximum number of iterations to converge.
#### Hyperparameter Tuning using HyperDrive
- The following steps were performed for effective hyperparameter tuning using the HyperDrive:
1. **Defining the parameter search space:**
   - A search space was specified to explore a range of values defined for each hyperparameter
```
search_space = { 
	"--C" : uniform(0.01, 1),
    	"--max_iter" : choice(100, 200, 300, 400, 500) 
}
```
1. **Sampling the Hyperparameter space:**
   - To find values over the specified search space a sampling method was used.
   - _This project uses the Random Sampling method to choose values from the mentioned search space_
```
ps = RandomParameterSampling (
	search_space
)
```  
1. **Specifying a primary metric:**
   - A primary metric is specified to optimize hyperparameter tuning. 
   - Each training run is evaluated for the primary metric. 
```
primary_metric = 'accuracy'
metric_goal = PrimaryMetricGoal.MAXIMIZE
```
1. **Specifying early termination policy:**
   - Early termination policy forces low performing runs to automatically terminate so that resources are not wasted on runs that will not give potential results
   - `Bandit Policy` used in this project is based on a _slack factor/slack amount_ and an _evaluation interval_. A run is terminated if the primary metric is not within the slack amount compared to the best performing run
```
policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1, delay_evaluation=5)
```
1. **Allocating resources**
   - Resources are allocated and controlled for running the hyperparameter tuning experiment
```
max_runs = 20
max_concurrent_runs = 4
```
1. **Using an Estimator**
   - The estimator is used to begin the training and invoke the script file.
```
est = SKLearn (
    source_directory= os.path.join("./"),
    compute_target= cpu_cluster,
    entry_script= "train.py"
)
```
1. **Configuring the experiment**
   - The attributes defined in the previous steps are specified to configure the hyperparameter tuning experiment before submitting the run using HyperDriveConfig()
1. **Submitting the experiment**
  - Experiment act an entry point for model runs. The run is submitted when submit() is invoked by the experiment object
```
hyperdrive_run = experiment.submit(hyperdrive_config)
```
1. **Visualizing the training runs**
   - The Notebook widget is used to visualize the experiment, run progress, logs and model metrics.
```
RunDetails(hyperdrive_run).show()
```
1. **Selecting the configurations of the best model**
   - Once all of the hyperparameter tuning runs completes, the best performing configuration and hyperparameter values are obtained
```
best_run = hyperdrive_run.get_best_run_by_primary_metric()
```
- **What are the benefits of the parameter sampler you chose?**
  - Random Sampling works with both discrete and continous search space unlike Grid Sampling. It also supports early termination policy unlike Bayesian Sampling. Hence Random Sampler helps in performing trial and error with values chosen over the search space and then refine the search space to obtain best results.
- **What are the benefits of the early stopping policy you chose?**
  - Early stopping helps in avoiding unnecessary usage of resources assigned to runs that performs poorly. This is ensured by terminating runs whose primary metric is not within the slack amount specified by the policy. 

### AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline Comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future Work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of Cluster Clean Up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
