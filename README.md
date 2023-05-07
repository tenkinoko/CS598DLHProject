# Reproduction Of GAMENet

This repository is the reproducibility project for CS598 Deep Learning for Healthcare in Spring 2023 at UIUC.

## Citation

The content in this repository is mainly taken from the [repository](https://github.com/sjy1203/GAMENet) from the original author of the [paper]([[1809.01852\] GAMENet: Graph Augmented MEmory Networks for Recommending Medication Combination (arxiv.org)](https://arxiv.org/abs/1809.01852)) GAMENet: Graph Augmented MEmory Networks for Recommending Medication Combination.

## Overview
This repository contains code necessary to run GAMENet model. GAMENet is an end-to-end model mainly based on graph convolutional networks (GCN) and memory augmented nerual networks (MANN). Paitent history information and drug-drug interactions knowledge are utilized to provide safe and personalized recommendation of medication combination. GAMENet is tested on real-world clinical dataset [MIMIC-III](https://mimic.physionet.org/) and outperformed several state-of-the-art deep learning methods in heathcare area in all effectiveness measures and also achieved higher DDI rate reduction from existing EHR data.

Certain adaptions have been made to meet the course requirements.


## Requirements
```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip3 install dill
```
- Pytorch >=0.4
- Python >=3.5 <=3.7


## Running the code
### Data preprocessing
In ./data, you can find the well-preprocessed data in pickle form. Also, it's easy to re-generate the data as follows:
1.  download [MIMIC data](https://mimic.physionet.org/gettingstarted/dbsetup/) and put DIAGNOSES_ICD.csv, PRESCRIPTIONS.csv, PROCEDURES_ICD.csv in ./data/
2.  download [DDI data](https://www.dropbox.com/s/8os4pd2zmp2jemd/drug-DDI.csv?dl=0) and put it in ./data/
3.  run code **./data/EDA.ipynb**

Data information in ./data:
  - records_final.pkl is the input data with four dimension (patient_idx, visit_idx, medical modal, medical id) where medical model equals 3 made of diagnosis, procedure and drug.
  - voc_final.pkl is the vocabulary list to transform medical word to corresponding idx.
  - ddi_A_final.pkl and ehr_adj_final.pkl are drug-drug adjacency matrix constructed from EHR and DDI dataset.
  - drug-atc.csv, ndc2atc_level4.csv, ndc2rxnorm_mapping.txt are mapping files for drug code transformation.


### Training and comparison
 Traning codes can be found in ./code/baseline/

 - **Nearest** will simply recommend the same combination medications at previous visit for current visit.
 - **Logistic Regression (LR)** is a logistic regression with L2 regularization. Here we represent the input data by sum of one-hot vector. Binary relevance technique is used to handle multi-label output.
 - **Leap** is an instance-based medication combination recommendation method.
 - **RETAIN** can provide sequential prediction of medication combination based on a two-level neural attention model that detects influential past visits and significant clinical variables within those visits.
 - **DMNC** is a recent work of medication combination prediction via memory augmented neural network based on differentiable neural computers. 


 ### GAMENet
 ```
 python train_GAMENet.py --model_name GAMENet --ddi# training with DDI knowledge
 python train_GAMENet.py --model_name GAMENet --ddi --resume_path Epoch_{}_JA_{}_DDI_{}.model --eval # testing with DDI knowledge
 python train_GAMENet.py --model_name GAMENet # training without DDI knowledge
 python train_GAMENet.py --model_name GAMENet --resume_path Epoch_{}_JA_{}_DDI_{}.model --eval # testing with DDI knowledge
 ```

## Results

Our model reproduces the following performance:

| Methods       | DDI Rate     | $\Delta$ DDI Rate % | Jaccard      | PR-AUC       | F1           |
| ------------- | ------------ | ------------------- | ------------ | ------------ | ------------ |
| LR            | 0.0832       | +2.59%              | 0.4792       | 0.7533       | 0.6331       |
| Leap          | 0.0586       | -27.74%             | 0.4571       | 0.6359       | 0.6088       |
| Retain        | 0.0808       | -0.37%              | 0.4897       | 0.7515       | 0.6121       |
| DMNC          | 0.0998       | +23.05%             | 0.4961       | 0.7699       | 0.6602       |
| ***GAMENet*** | ***0.0788*** | ***-2.80%***        | ***0.5164*** | ***0.7709*** | ***0.6723*** |

Comparing to the original performance:

| Methods       | DDI Rate     | $\Delta$ DDI Rate % | Jaccard      | PR-AUC       | F1           |
| ------------- | ------------ | ------------------- | ------------ | ------------ | ------------ |
| LR            | 0.0786       | +1.16%              | 0.4075       | 0.6716       | 0.5658       |
| Leap          | 0.0532       | -31.53%             | 0.3844       | 0.5501       | 0.5410       |
| Retain        | 0.0797       | +2.57%              | 0.4168       | 0.6620       | 0.5781       |
| DMNC          | 0.0949       | +22.14%             | 0.4343       | 0.6856       | 0.5934       |
| ***GAMENet*** | ***0.0749*** | ***-3.60%***        | ***0.4509*** | ***0.6904*** | ***0.6081*** |

