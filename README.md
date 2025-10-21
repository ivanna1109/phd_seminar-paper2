# PhD Studies - Subject: Seminar Paper 2

## Topic - Multi-class Prediction of Estrogenic Receptor Activity Using Graph Neural Networks and Analysis of Global Features

### Overview

This repository contains the code and experimental results for a project focused on applying **Graph Neural Networks (GNNs)** to the problem of **multi-class classification** of **Estrogenic Receptor (ESR) activity**.

The primary goal is to classify novel small molecules into three activity classes: **Inactive, Active on ESR1/ER$\alpha$**, and **Dual Active (active both on ESR1 and ESR2)**. 
A crucial part of this work is the detailed analysis of the impact of incorporating additional **global molecular descriptors (features)** on the models' predictive performance.

Four key GNN architectures were implemented and evaluated: **GCN, GraphSAGE, GAT, and GIN**. Advanced techniques, including graph isomorphism-based data augmentation and hyperparameter optimization (Optuna), were also employed.

### Repository Contents
* **`data_preprocessing/`**
    * Contains scripts responsible for **processing raw data** (from $\texttt{.csv}$ files), conducting **(EDA)**, data transformation, visualization, and creating $\text{TFRecord}$ files using **graph isomorphism augmentation**.
    * **`dataset/`**: Original, raw data.
    * **`dataset_for_multi/`**: Contains files for creating the dataset for the multi-classification problem.
    * **`datasets`**: Final $\text{TFRecords}$ of data later used in the training process for GNNs.

* **`training/`**
    * Contains files related to the multi-class training process.
    * **`x_train.py`**: Scripts related to the training process of different models.
    * **`hyperparam_x.py`**: Scripts related to $\text{Optuna}$ optimization for different models.
    * **`initial_results/`**: Contains initial results gained in the initial training process.
    * **`jobs/`**: Example shell scripts ($\texttt{.sh}$) for submitting batch jobs.
    * **`xxxx_logs/`**: Output files for job status and monitoring.

* **`metrics/`**: Contains definitions for metrics ($\text{F1-score, weighted-F1, balanced-accuracy}$) used in evaluation.

* **`models/`**: Definitions of every utilized model ($\text{GCN, GraphSAGE, GAT, GIN}$), including implementations with and without **global molecular features**.

* **`multi_spektral/`**: Contains scripts for converting loaded data into the **Spektral dataset format**, needed for GNN models.

* **`optuna/`**: Results of $\text{Optuna}$ hyperparameter optimization for every model ($\texttt{.csv}$ files for trial records and a final $\texttt{.txt}$ file of the best set of hyperparameter values).

* **`final_results/`**: Contains final training results of every model used on the dataset **without** global molecular features.

* **`results_with_features/`**: Contains final training results of every model used on the dataset **containing global molecular features**.

* **`output_plots_xxxxx/`**: Contains final plot results of every model used on datasets with and without global molecular features.

* **`visualize_results.ipynb`**: File related to creating plots and visualizations based on the $\texttt{.csv}$ results of every model.
