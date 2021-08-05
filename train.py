# This material was prepared as an account of work sponsored by an agency of the 
# United States Government. Neither the United States Government nor the United 
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
#                    PACIFIC NORTHWEST NATIONAL LABORATORY
#                               operated by
#                                BATTELLE
#                                for the
#                      UNITED STATES DEPARTMENT OF ENERGY
#                       under Contract DE-AC05-76RL01830

import os
import torch
from torch import optim, nn
from torch.utils.data.dataloader import DataLoader
from model import Generator
from utils import SDataset
#torch.autograd.set_detect_anomaly(True)
from datetime import datetime
import time
import csv
import argparse

parser = argparse.ArgumentParser()
## directories
parser.add_argument('--data', type=str, default='./processed/train.npy', help='path to processed training data')
parser.add_argument('--save_dir', type=str, default='./results/', help='path to save') 
## dataset params
parser.add_argument('--n_atoms', type=int, default=19, help='number of atoms in system')
parser.add_argument('--window_size', type=int, default=10, help='window size')
## training params
parser.add_argument('--epochs', type=int, default=5000, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=499, help='batch size, must be a factor of dataset size')
parser.add_argument('--in_features', type=int, default=1, help='input features')
parser.add_argument('--out_features', type=int, default=64, help='output features for GCN')
parser.add_argument('--lstm_features', type=int, default=128, help='output features for LSTM')
parser.add_argument('--optimizer', type=str, default='adam', help='optimizer: adam or rmsprop')
parser.add_argument('--learning_rate', type=float, default=0.005, help='learning rate')
parser.add_argument('--gradient_clip', type=float, default=5.0, help='gradient clipping')
args = parser.parse_args()

# Make the save directory
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# Create data loader
train_data = SDataset(args.data)
print(f'data loaded from {args.data}')
print(f'total samples in training set: {train_data.__len__()}')

train_loader = DataLoader(
    dataset=train_data,
    batch_size=args.batch_size,
    shuffle=True,
    pin_memory=True
)

generator = Generator(
    window_size=args.window_size,
    node_num=args.n_atoms,
    in_features=args.in_features,
    out_features=args.out_features,
    lstm_features=args.lstm_features
)


generator = generator.cuda()

mse = nn.MSELoss(reduction='mean') # could choose 'sum'

# choose optimization algorithm 
if args.optimizer == 'rmsprop':
    train_optimizer = optim.RMSprop(generator.parameters(), lr=args.learning_rate)
elif args.optimizer == 'adam':
    train_optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate)
else:
    print('optimizer not supported: choose adam or rmsprop')

start = datetime.now()
print(f'training started at {start}')

for epoch in range(args.epochs):
    for i, data in enumerate(train_loader):
        st = time.time()
        train_optimizer.zero_grad()
        # separate the data into window and t+1
        in_shots, out_shot = data
        in_shots, out_shot = in_shots.cuda(), out_shot.unsqueeze(1).cuda()
        # use window to predict t+1
        predicted_shot = generator(in_shots)
        out_shot = out_shot.view(args.batch_size, -1)
        # compare actual t+1 and predicted t+1
        loss = mse(predicted_shot, out_shot.unsqueeze(0))
        # update weights
        loss.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), args.gradient_clip)
        train_optimizer.step()
        end = time.time()
        # write to log file
        if epoch == 0 and i == 0:
            with open(os.path.join(args.save_dir,'log-train.csv'), mode='w') as f:
                writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["epoch","step","loss","time"])
                writer.writerow([epoch, i, loss.item(), end-st])
        else:
            with open(os.path.join(args.save_dir,'log-train.csv'), mode='a') as f: 
                writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([epoch, i, loss.item(), end-st]) 

end = datetime.now()
print(f'training finished after {(end-start).total_seconds():0.2f} s')
            
# save trained model
torch.save(generator, os.path.join(args.save_dir, 'generator.pkl'))
print(f'generator saved at {os.path.join(args.save_dir, "generator.pkl")}')
