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

import torch
from torch import nn
from torch.nn import init

class GraphConvolution(nn.Module):

    def __init__(self, window_size, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weights = nn.Parameter(
            torch.Tensor(window_size,in_features, out_features)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.weights)

    def forward(self, adjacency, nodes):
        """
        :param adjacency: FloatTensor (batch_size, window_size, node_num, node_num)
        :param nodes: FloatTensor (batch_size, window_size, node_num, in_features)
        :return output: FloatTensor (batch_size, window_size, node_num, out_features)
        """
        batch_size = adjacency.size(0)
        window_size, in_features, out_features = self.weights.size()
        weights = self.weights.unsqueeze(0).expand(batch_size, window_size, in_features, out_features)
        output = adjacency.matmul(nodes).matmul(weights)
        return output

class Generator(nn.Module):

    def __init__(self, window_size, node_num, in_features, out_features, lstm_features):
        super(Generator, self).__init__()
        self.window_size = window_size
        self.node_num = node_num
        self.in_features = in_features
        self.out_features = out_features
        self.gcn = GraphConvolution(window_size, in_features, out_features)
        self.lstm = nn.LSTM(
            input_size=out_features * node_num,
            hidden_size=lstm_features,
            num_layers=1,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(lstm_features, node_num * node_num),
            nn.Tanh()
        )

    def forward(self, in_shots):
        """
        :param in_shots: FloatTensor (batch_size, window_size, node_num, node_num)
        :return out_shot: FloatTensor (batch_size, node_num * node_num)
        """
        batch_size, window_size, node_num = in_shots.size()[0: 3]
        eye = torch.eye(node_num).cuda().unsqueeze(0).unsqueeze(0).expand(batch_size, window_size, node_num, node_num)
        in_shots = in_shots + eye
        diag = in_shots.sum(dim=-1, keepdim=True).pow(-0.5).expand(in_shots.size()) * eye
        adjacency = diag.matmul(in_shots).matmul(diag)
        nodes = torch.rand(batch_size, window_size, node_num, self.in_features).cuda()
        gcn_output = self.gcn(adjacency, nodes)
        gcn_output = gcn_output.view(batch_size, window_size, -1)
        _, (hn, _) = self.lstm(gcn_output)
        output = self.ffn(hn)
        return output

