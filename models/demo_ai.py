import torch_geometric.nn as nn
import torch
import torch_geometric as tg
import torch.nn.functional as F
import pytorch_lightning as L


class Demo_ai(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.in_channels = int(kwargs["input_features"])

        self.lin1 = nn.Linear(self.in_channels, 10)
        self.conv1 = nn.conv.SAGEConv(10, 10)
        self.conv2 = nn.conv.SAGEConv(10, 10)
        self.lin2 = nn.Linear(10, 3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.lin2(x)
        return x
