# Molecular Dynamic Graph Neural Network

We apply a temporal edge prediction model for weighted dynamic graphs to predict time-dependent changes in molecular structure. Molecules are represented as graphs, where vertices represent atoms, and edges represent euclidian distance between atom pairs. We use a subset of molecular trajectories from the ISO17 dataset. For more information, see http://quantum-machine.org/datasets/.

## Getting Started
* run "get_data.sh" bash script. This should download the ISO17 database into your working directory.
* run "preprocess.py" python script. This creates the matrices for training and test sets. 
* run "train.py" python script. This train over the training set and save your model as a "generator.pkl" file. 
* run "test.py" python script. Will evalute trained model over test set and save results as a csv file.

For example:
```
./get_data.sh #run bash script
python preprocess.py 
python train.py
python test.py

```

## Finetuning
It is inefficient to train a network from scratch on a large number of snapshots to predict the subsequent trajectory of new molecules. Our proposed solution is to finetune our trained networks for a reduced number of epochs on a small number of new snapshots. We can use the weights from a pretrained isomer network as the initial weights to train a new network on a new isomer. This greatly reduces the training time while keeping the accuracy of the results. 

`python finetune.py --model path_to_model --data path_to_new_isomer_data`


## Customization

preprocessing, training, and testing all come with flags for additional customization. Add "--help" to the end of any script to view these flags and how to use them
`python train.py --help`

