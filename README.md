# PhD Studies - Subject: Seminar Paper 2

## Topic - Multi-class Prediction of Estrogenic Receptor Activity Using Graph Neural Networks and Analysis of Global Features

### Overview

This repository contains the code and experimental results for a project focused on applying **Graph Neural Networks (GNNs)** to the problem of **multi-class classification** of **Estrogenic Receptor (ESR) activity**.

The primary goal is to classify novel small molecules into three activity classes: **Inactive, Active on ESR1/ER$\alpha$**, and **Dual Active (active both on ESR1 and ESR2)**. 
A crucial part of this work is the detailed analysis of the impact of incorporating additional **global molecular descriptors (features)** on the models' predictive performance.

Four key GNN architectures were implemented and evaluated: **GCN, GraphSAGE, GAT, and GIN**. Advanced techniques, including graph isomorphism-based data augmentation and hyperparameter optimization (Optuna), were also employed.

### Repository Contents
 | :--- | :--- |
| **`data_preprocessing/`** | Contains all scripts for **raw data processing** (from $\texttt{.csv}$), Exploratory Data Analysis ($\text{EDA}$), data transformation, and creating $\text{TFRecord}$ files using **graph isomorphism augmentation**. |
| `data_preprocessing/dataset/` | Original, raw data files. |
| `data_preprocessing/dataset_for_multi/` | Files specifically used for generating the multi-class dataset. |
| `data_preprocessing/datasets/` | Final $\text{TFRecord}$ files of the processed data used for GNN training. |
| **`training/`** | Primary directory containing scripts related to model training and hyperparameter optimization. |
| `training/x_train.py` | Scripts responsible for the training process of the various GNN models. |
| `training/hyperparam_x.py` | Scripts used for $\text{Optuna}$ hyperparameter optimization runs. |
| `training/initial_results/` | Contains the initial results obtained during the early stages of the training process. |
| `training/jobs/` | Example shell scripts ($\texttt{.sh}$) for submitting batch processing jobs. |
| `training/xxxx_logs/` | Output files containing job status and monitoring logs. |
| **`models/`** | Python files containing the definition and architecture implementation of every utilized GNN model: **GCN, GraphSAGE, GAT, and GIN** (both with and without global molecular features). |
| **`multi_spektral/`** | Scripts necessary for converting the loaded molecular data into the **Spektral dataset format**, which is required for GNN model implementation. |
| **`metrics/`** | Contains definitions for custom evaluation metrics ($\text{F1-score, weighted-F1, balanced-accuracy}$) used in the project. |
| **`optuna/`** | Results from the $\text{Optuna}$ optimization process, including $\texttt{.csv}$ files for trial records and $\texttt{.txt}$ files detailing the final best hyperparameter set for each model. |
| **`final_results/`** | Contains the final training and evaluation results for every model used on datasets **without** global molecular features. |
| **`results_with_features/`** | Contains the final training and evaluation results for every model used on datasets **containing global molecular features**. |
| **`output_plots_xxxxx/`** | Contains the final plot results (training curves, performance visualizations) for models trained with and without global features. |
| **`visualize_results.ipynb`** | Jupyter Notebook file dedicated to creating plots and visualizations based on the collected $\texttt{.csv}$ results from all model evaluations. |
