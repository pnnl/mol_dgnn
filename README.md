# Molecular Dynamic Graph Neural Network

We apply a temporal edge prediction model for weighted dynamic graphs to predict time-dependent changes in molecular structure. Molecules are represented as graphs, where vertices represent atoms, and edges represent euclidian distance between atom pairs. We use a subset of molecular trajectories from the ISO17 dataset. For more information, see http://quantum-machine.org/datasets/.

## Getting Started
* run "get_data.sh" bash script. This should download the ISO17 database into your working directory.
* run "preprocess.py" python script. This creates the matrices for training and test sets. 
* run "train.py" python script. This train over the training set and save your model as a "generator.pkl" file. 
* run "test.py" python script. Will evalute trained model over test set and save results as a csv file.



