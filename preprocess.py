# This material was prepared as an account of work sponsored by an agency of the
# United States Government.  Neither the United States Government nor the United
# States Department of Energy, nor Battelle, nor any of their employees, nor any
# jurisdiction or organization that has cooperated in the development of these
# materials, makes any warranty, express or implied, or assumes any legal
# liability or responsibility for the accuracy, completeness, or usefulness or
# any information, apparatus, product, software, or process disclosed, or
# represents that its use would not infringe privately owned rights. Reference
# herein to any specific commercial product, process, or service by trade name,
# trademark, manufacturer, or otherwise does not necessarily constitute or imply
# its endorsement, recommendation, or favoring by the United States Government
# or any agency thereof, or Battelle Memorial Institute. The views and opinions
# of authors expressed herein do not necessarily state or reflect those of the
# United States Government or any agency thereof.
#                 PACIFIC NORTHWEST NATIONAL LABORATORY
#                            operated by
#                             BATTELLE
#                             for the
#                   UNITED STATES DEPARTMENT OF ENERGY
#                    under Contract DE-AC05-76RL01830

import os
import numpy as np
import utils
import networkx as nx
import sys
from ase.db import connect
import argparse

parser = argparse.ArgumentParser()
## directories
parser.add_argument('--data', type=str, default='./iso17/test_other.db', help='path to database')
parser.add_argument('--save_dir', type=str, default='./processed/', help='path to save data')
## train/test parameters
parser.add_argument('--N', type=float, default=10.0, help='normalization constant')
parser.add_argument('--window_size', type=int, default=10, help='window size')
parser.add_argument('--split', type=int, default=0.8, help='percent of data for training set')
parser.add_argument('--seed', type=int, default=42, help='seed for creating the train/test split')
args = parser.parse_args()

# set np.random seed
np.random.seed(args.seed)

# create save dir if it does not exist
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# collect data from ase database
#structures=[]
dft,atom_list,distance_matrix=[],[],[]
with connect(args.data) as conn:
   for row in conn.select():
       #structures.append(row.toatoms().get_positions())
       dft.append(row['total_energy'])
       atom_list.append(row.toatoms().get_atomic_numbers())
       distance_matrix.append(row.toatoms().get_all_distances())
print(f'Data loaded from {args.data}')

# save all data
np.save(os.path.join(args.save_dir,'adjacency_matrix.npy'), np.array(distance_matrix), allow_pickle=True)    
np.save(os.path.join(args.save_dir,'atom_list.npy'), np.array(atom_list), allow_pickle=True)
np.save(os.path.join(args.save_dir,'dft_energy.npy'), np.array(dft), allow_pickle=True)

# create training and test sets

# normalize
A_mats = [snapshot/args.N for snapshot in distance_matrix]

# split into separate sets
# current code specific to ISO17 test_other.db 
# each set has 5000 snapshots and are collected in order from the db
A_mats = np.array([A_mats[i:i+5000] for i in range(0,len(A_mats),5000)]) 
sets = A_mats.shape[0]

# split data into windows
split_sets = np.array([utils.split_data(A_mats[i,:,:,:], args.window_size) for i in range(sets)])

total_num = split_sets.shape[1]
train_idx = int(args.split*total_num)

# shuffle along windows
split_sets = utils.shuffle_along_axis(split_sets, 1)

train_data =np.array(split_sets[:,:train_idx,:,:,:])
test_data = np.array(split_sets[:,train_idx:,:,:,:])

train_data = train_data.reshape(-1, *train_data.shape[2:])
test_data = test_data.reshape(-1, *test_data.shape[2:])
print(f'Shape of training data: {train_data.shape}')
print(f'Shape of test data: {test_data.shape}')

np.save(os.path.join(args.save_dir,'train.npy'), train_data, allow_pickle=True)
np.save(os.path.join(args.save_dir,'test.npy'), test_data, allow_pickle=True)

print(f'Data saved at {args.save_dir}')
