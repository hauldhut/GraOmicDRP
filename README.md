# Resources
+ README.md: this file.
+ Datasets: 
    + GDSC, TCGA and PDX
    + Download from: https://drive.google.com/drive/folders/1CKswGNVdlRupZIAUw3yKyqkSn0NNhdyr?usp=sharing
	
# Source code
### Regression model
+ Single -omic data
	+ GE
    + MUT
    + METH
+ Multiple -omic data
	+ GE_MUT_METH
	+ GEN_MUT
	+ GE_METH
    + MUT_METH
	
### Classification model (To compare with MOLI method)
+ Multiple -omic data
	+ GE_MUT_METH
	+ GEN_MUT
	+ GE_METH
	+ MUT_METH
	
# Dependencies
+ Torch
+ Pytorch_geometric
+ Rdkit
+ Matplotlib
+ Pandas
+ Numpy
+ Scipy

# Step-by-step running
## Regression model:
### Create data
`python preprocess.py --choice 0`

choice:       0: create mixed test dataset       1: create saliency map dataset       2: create blind drug dataset       3: create blind cell dataset

This returns file pytorch format (.pt) stored at data/processed including training, validation, test set.

### Train model
`python training.py --model 0 --train_batch 1024 --val_batch 1024 --test_batch 1024 --lr 0.0001 --num_epoch 300 --log_interval 20 --cuda_name "cuda:0"`

model:       1: GINConvNet       2: GATNet       3: GAT_GCN       4: GCNNet

To train a model using training data. The model is chosen if it gains the best MSE for testing data.

This returns the model and result files for the modelling achieving the best MSE for testing data throughout the training.2

## Classification model:
### Create data
`python preprocess.py --choice 0`
### Train model 
+ Need to run Regession model first to get the weight for the feature extraction task
+ Run end to end jupyter notebook file (remember change the correct data path)



