# PhD Studies - Subject: Seminar Paper 2

## Topic - Multi-class Prediction of Estrogenic Receptor Activity Using Graph Neural Networks and Analysis of Global Features

### Overview

This repository contains the code and experimental results for a project focused on applying **Graph Neural Networks (GNNs)** to the problem of **multi-class classification** of **Estrogenic Receptor (ESR) activity**.

The primary goal is to classify novel small molecules into three activity classes: **Inactive, Active on ESR1/ER$\alpha$**, and **Dual Active (active both on ESR1 and ESR2)**. 
A crucial part of this work is the detailed analysis of the impact of incorporating additional **global molecular descriptors (features)** on the models' predictive performance.

Four key GNN architectures were implemented and evaluated: **GCN, GraphSAGE, GAT, and GIN**. Advanced techniques, including graph isomorphism-based data augmentation and hyperparameter optimization (Optuna), were also employed.

### Repository Contents
  * **`data_preprocessing/`**
     * Contains scripts responsible for **processing raw data** (from CSV files), **(EDA)**, data transformation and visualization, and creating the TFRecords using an **augmentation technique (graph isomorphism)**
    **`dataset/`** - Original, raw data
    **`dataset_for_multi/`** - contains files for creating dataset for multi-classification problem
    **`datasets`** - final TFRecords of data later used in training process for GNNS
    **`metrics/`** - contains some metrics definition (f1-score, weighted-f1, balanced-accuracy)
    **`models/`** - definition of every utilized model (gcn, gsage, gat, gin), with global features and without
    **`optuna/`** - results of optuna hyperparam optimization for every model (cvs files for every set of hyperparam values used in trial, for every model), and final txt file of the best set of hyperparam values
    **`multi_spektral/`** - contains scripts for converting loaded data into spektral dataset, needed for GNN models
    **`final_results/`** - contains final training results of every model used on dataset containing global molecular features
    **`results_with_features/`** - contains final training results of every model used on dataset containing global molecular features
    **`output_plots_xxxxx/`** - contains final plot results of every model used on dataset with and without global molecular features
  * **`logs/`** - few output files for job status and monitoring
  * **`training/`** - files related to binary classification: ESR1/ESR2
     *  **`x_train.py`** - files related to diff models training process
     *  **`hyperparam_x.py`** - files related to optuna optimization for diff models
     *  **`initial_results/`** - contains initial results gained in initial training process
     *  **`jobs/`** - Example shell scripts (`.sh`) for submitting batch jobs
     *  **`xxxx_logs/`** - few output files for job status and monitoring
  * **`visualize_results.ipynb/`** - file related to plots creation based on results in csv format of every model
