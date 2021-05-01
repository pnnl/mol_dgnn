import os
import torch
from torch.utils.data.dataloader import DataLoader
from utils import SDataset
from utils import MSE, EdgeWiseKL, MissRate
import csv
import numpy as np
import argparse

parser = argparse.ArgumentParser()
## directories
parser.add_argument('--data', type=str, default='./processed/test.npy', help='processed data file for test set')
parser.add_argument('--save_dir', type=str, default='./results/', help='path to save')
## dataset params
parser.add_argument('--n_atoms', type=int, default=19, help='number of atoms in system')
parser.add_argument('--window_size', type=int, default=10, help='window size')
## model params
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--model', type=str, default='./results/generator.pkl', help='path to pretrained model')
args = parser.parse_args()

# Make the save directory
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

save_path = args.save_dir
# Create data loader
test_data = SDataset(args.data)
print(f'data loaded from {args.data}')
print(f'total samples in training set: {test_data.__len__()}')

test_loader = DataLoader(
    dataset=test_data,
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=True
)

# load pretrained model
generator = torch.load(args.model)
generator = generator.cuda()
print(f'model loaded from {args.model}')

total_samples = 0
total_mse = 0
total_kl = 0
total_missrate = 0

for i, data in enumerate(test_loader):
    in_shots, out_shot = data
    in_shots, out_shot = in_shots.cuda(), out_shot.cuda()
    predicted_shot = generator(in_shots)
    predicted_shot = predicted_shot.view(-1, args.n_atoms, args.n_atoms)
    # average to make symmetric Aij = Aji
    predicted_shot = (predicted_shot + predicted_shot.transpose(1, 2)) / 2
    # put 0 on the diagnal (no self-loops)
    for j in range(args.n_atoms):
        predicted_shot[:, j, j] = 0
    batch_size = in_shots.size(0)
    total_samples += batch_size
    total_mse += batch_size * MSE(predicted_shot, out_shot)
    total_kl += batch_size * EdgeWiseKL(predicted_shot, out_shot)
    total_missrate += batch_size * MissRate(predicted_shot, out_shot)

    predicted_shot = predicted_shot.cpu().detach().numpy()
    out_shot = out_shot.cpu().detach().numpy()
    if i == 0:
        all_predicted_shots = predicted_shot
        all_true_shots = out_shot
    else:
        all_predicted_shots = np.vstack((all_predicted_shots, predicted_shot))
        all_true_shots = np.vstack((all_true_shots, out_shot))

# save true and predicted normalized adjacency matrices
np.save(os.path.join(save_path, 'true.npy'), all_true_shots)
np.save(os.path.join(save_path, 'pred.npy'), all_predicted_shots)


# print and save statistics
print('MSE: %.4f' % (total_mse / total_samples))
print('edge wise KL: %.4f' % (total_kl / total_samples))
print('miss rate: %.4f' % (total_missrate / total_samples))

with open(os.path.join(save_path, f'test_set_statistics.csv'), mode='a') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Dataset','Model','MSE','Edge-wise KL','Miss rate'])
    writer.writerow([args.data, args.model,(total_mse / total_samples),(total_kl / total_samples),(total_missrate / total_samples)])
