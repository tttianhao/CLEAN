# CLEAN: Enzyme Function Prediction using Contrastive Learning

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache-2.svg)](https://opensource.org/licenses/Apache-2.0)

This is the official repository for the paper *Enzyme Function Prediction using Contrastive Learning*. **CLEAN**, **Contrastive Learning enabled Enzyme ANnotation**, is a machine learning algorithm to assign Enzyme Commission (EC) number with better accuracy, reliability, and sensitivity than all existing computational tools. 

To use CLEAN to inference the EC number for any amino acid sequence, we included the pretrained weights for both the 70% and 100% identity clustering split of SwissProt (expertly reviewed portion of the UniProt, total ~220k training data). User can follow the instruction below on how to install and inference with CLEAN. We also provide full training scripts.
<p align="center">
<img src="CLEAN.png" alt="drawing" width="600"/>
</p>

## Install
### Requirements
Python >= 3.6; PyTorch >= 1.11.0; CUDA >= 10.1
### Procedures
Install requirement and build CLEAN
```python
pip install -r requirements.txt
git clone https://github.com/facebookresearch/esm.git
python build.py install
```
Next, esm-1b embeddings need to be pre-computed from a FASTA file. There are two options:
1. Retrive all embedding for all SwissProt sequences (slow, but required for training)
2. Retrive only embeddings for enzymes to be inferenced (fast)

For option 1, run following commands in python:
```python
python
>>> from CLEAN.utils import *
>>> ensure_dirs("data/esm_data")
>>> ensure_dirs("data/pretrained")
>>> csv_to_fasta("data/split100.csv", "data/split100.fasta")
>>> retrive_esm1b_embedding("split100")
```
For option 2, move the fasta file (for example, `test.fasta`) to be inferred to `/data`, and run following commands :
```python
python
>>> from CLEAN.utils import *
>>> ensure_dirs("data/esm_data")
>>> ensure_dirs("data/pretrained")
>>> retrive_esm1b_embedding("test")
```
## Inference
### Preparation 
We included pretrained weights for 70% and 100% splits, along with pre-evaluated embeddings for each EC cluster centers for fastest inference. Download, unzip [these files](https://drive.google.com/file/d/1zrEU-HPNV3wp7wLAx4KnuiVyD794Oboj/view?usp=sharing) and move to `data/pretrained`.

We offer two EC-calling inference algorithms: `max-separation` and `p-value`. `max-separation` consistently gives better precision and recall, but results from `p-value` can be controlled by adjusting `p_value` as a hyperparameter. 

Before inference, AA sequences to be inferred are stored in a CSV file, with the same format as the `split100.csv`. **The field `EC number` in the csv file can be any EC number if unknow, but please ignore the printed evaluation metrics in this case.** The esm-1b embeddings of the infered sequences need to be pre-computed using the following commands (using `new.csv` as an example):
```python
python
>>> from CLEAN.utils import *
>>> csv_to_fasta("data/new.csv", "data/new.fasta")
>>> retrive_esm1b_embedding("new")
```
### Inference with p-value 


For inferencing using `p-value`, there are two hyperparameter: `nk_random` and `p_value`. `nk_random` is the number of randomly chosen enzymes (in thousands) from the training set used for calculating background distances (distances to incorrect EC numbers) for each EC number. `p-value` is the threshould for a EC number to be considered significant relative to the backgound distances. The following commands show how to get EC prediction results from `p-value`:
```python
python
>>> from CLEAN.infer import infer_pvalue
>>> train_data = "split100"
>>> test_data = "new"
>>> infer_pvalue(train_data, test_data, p_value=1e-5, nk_random=20, report_metrics=True)
```
This should produce following results:
```
The embedding sizes for train and test: torch.Size([241025, 128]) torch.Size([392, 128])
Calculating eval distance map, between 392 test ids and 5242 train EC cluster centers
############ EC calling results using random chosen 20k samples ############
---------------------------------------------------------------------------
>>> total samples: 392 | total ec: 177
>>> precision: 0.558 | recall: 0.477 | F1: 0.482 | AUC: 0.737  
---------------------------------------------------------------------------
```

### Inference with max-separation 
For inferencing using `max-separation`, there are no hyperparameters to tune: it's a greedy approach that prioritizes EC numbers that have the maximum separation to other EC numbers in terms of the pairwise distance to the query sequence. `max-separation` gives a deterministic prediction and usually outperforms `p-value` in turns of precision and recall. Because this algorithm does not need to sample from the training set, it is much faster than `p-value`. The following commands show how to get EC predicition results from `max-separation`:
```python
python
>>> from CLEAN.infer import infer_maxsep
>>> train_data = "split100"
>>> test_data = "new"
>>> infer_maxsep(train_data, test_data, report_metrics=True)
```
This should produce following results:
```
The embedding sizes for train and test: torch.Size([241025, 128]) torch.Size([392, 128])
Calculating eval distance map, between 392 test ids and 5242 train EC cluster centers
############ EC calling results using maximum separation ############
---------------------------------------------------------------------------
>>> total samples: 392 | total ec: 177 
>>> precision: 0.596 | recall: 0.479 | F1: 0.497 | AUC: 0.739 
---------------------------------------------------------------------------
```
### Interpreting prediction csv file
The prediction results are store in the folder `results/` with file name = test_data + infer_algo (for example, `new_maxsep.csv`). An example output would be:
```
Q9RYA6,EC:5.1.1.20/7.4553
O24527,EC:2.7.11.1/5.8561
Q5TZ07,EC:3.6.1.43/8.0610,EC:3.1.3.4/8.0627,EC:3.1.3.27/8.0728
```
Where the first column (`Q9RYA6`) is the id of the enzyme, second column (`EC:5.1.1.20/7.4553`) is the predicted EC number and pairwise distance between cluster center of `5.1.1.20` and `Q9RYA6`. Note in the case of enzyme `Q5TZ07`, three enzyme functions are predicted.


## Training
TO BE UPDATED
