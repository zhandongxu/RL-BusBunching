import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_scatter import scatter_sum
from torch_geometric.data import batch


def make_seq(in_size, out_size, hidde_size=(64, 64), output_activation="None") -> nn.Module:
    mods = []
    # input layer
    first_layer = torch.nn.Linear(in_size, hidde_size[0])
    mods.append(first_layer)
    mods.append(nn.ReLU())
    # hidden layer
    for i in range(len(hidde_size) - 1):
        hidden_layer = torch.nn.Linear(hidde_size[i], hidde_size[i + 1])
        mods.append(hidden_layer)
        mods.append(nn.ReLU())
    # output layer
    output_layer = torch.nn.Linear(hidde_size[-1], out_size)
    mods.append(output_layer)
    if output_activation == "None":
        pass
    if output_activation == "Sigmoid":
        mods.append(torch.nn.Sigmoid())
    if output_activation == "Softmax":
        mods.append(torch.nn.modules.activation.Softmax(dim=-1))
    return nn.Sequential(*mods)


class ActorNetwork(nn.Module):
    def __init__(self, state_size, out_size, hidde_size=(64, 64), output_activation="None"):
        super(ActorNetwork, self).__init__()
        self.__in_size = state_size
        self.__out_size = out_size
        self.__output_activation = output_activation
        self.__layers = make_seq(state_size-1, out_size, hidde_size, output_activation)

    def forward(self, x) -> Tensor:
        x = x[:, 0:2]
        x = self.__layers(x)
        return x


class CriticNetwork(nn.Module):
    def __init__(self, state_size, out_size, hidde_size=(64, 64), output_activation="None"):
        super(CriticNetwork, self).__init__()
        self.__in_size = state_size
        self.__out_size = out_size
        self.__output_activation = output_activation
        self.__layers = make_seq(state_size-1, out_size, hidde_size, output_activation)

    def forward(self, x) -> Tensor:
        x = x[:, 0:2]
        x = self.__layers(x)
        return x


class DuelingNetwork(nn.Module):
    def __init__(self, state_size, out_size, hidde_size=(64, 64)):
        super(DuelingNetwork, self).__init__()

        self.input_layer = nn.Linear(state_size-1, hidde_size[0])
        self.hidden_layer = nn.Linear(hidde_size[0], hidde_size[1])
        self.V = nn.Linear(hidde_size[1], 1)
        self.A = nn.Linear(hidde_size[1], out_size)

    def forward(self, x):
        x = x[:, 0:2]
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hidden_layer(x))

        V = self.V(x)
        A = self.A(x)

        Q = V + A - A.mean()
        return Q


class ActorCritic(nn.Module):
    def __init__(self, state_size, out_size, hidde_size=(64, 64)):
        super(ActorCritic, self).__init__()

        self.critic_linear1 = nn.Linear(state_size-1, hidde_size[0])
        self.critic_linear2 = nn.Linear(hidde_size[0], hidde_size[1])
        self.critic_linear3 = nn.Linear(hidde_size[1], 1)

        self.actor_linear1 = nn.Linear(state_size-1, hidde_size[0])
        self.actor_linear2 = nn.Linear(hidde_size[0], hidde_size[1])
        self.actor_linear3 = nn.Linear(hidde_size[1], out_size)

    def forward(self, state):
        state = state[:, 0:2]
        value = F.relu(self.critic_linear1(state))
        value = F.relu(self.critic_linear2(value))
        value = F.relu(self.critic_linear3(value))

        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.relu(self.actor_linear2(policy_dist))
        policy_dist = F.softmax(self.actor_linear3(policy_dist), dim=1)

        return value, policy_dist


class Event_Critic_Net(nn.Module):
    def __init__(self, state_size, hidden_size, out_size, init_type='default'):
        super().__init__()
        self.up_conv1 = GATConv(state_size, hidden_size, add_self_loops=False)
        # self.up_conv2 = GATConv(hidden_size, hidden_size)
        self.down_conv1 = GATConv(
            state_size, hidden_size, add_self_loops=False)
        # self.down_conv2 = GATConv(hidden_size, hidden_size)

        # self.up_conv1 = GCNConv(state_size, hidden_size)
        # self.up_conv2 = GCNConv(hidden_size, hidden_size)
        # self.down_conv1 = GCNConv(state_size, hidden_size)
        # self.down_conv2 = GCNConv(hidden_size, hidden_size)

        self.mlp = torch.nn.Linear(hidden_size, out_size)
        # if init_type == 'normal':
        #     torch.nn.init.normal_(self.mlp.weight)

    def forward(self, batch_up_data, batch_down_data):
        # # for upstream event graph
        # up_x, up_edge_index, up_batch = batch_up_data.x.to(torch.float32), batch_up_data.edge_index, batch_up_data.batch
        # unique_up_batches = torch.unique(up_batch)
        # batch_outputs = []
        # for unique_up_batch in unique_up_batches:
        #     mask = (up_batch == unique_up_batch)
        #     batch_up_x = up_x[mask]
        #     batch_up_edge_index = up_edge_index[:, mask]
        #     output = self.up_conv1(batch_up_x, batch_up_edge_index)
        #
        #     batch_outputs.append(output)
        #
        # up_output = torch.stack(batch_outputs)

        # for upstream event graph
        up_x, up_edge_index, up_batch = batch_up_data.x.to(torch.float32), batch_up_data.edge_index, batch_up_data.batch
        up_x = self.up_conv1(up_x, up_edge_index)

        # get the number of nodes in each graph in the batch
        up_num_nodes_per_graph = scatter_sum(up_batch.new_ones(
            batch_up_data.num_nodes), up_batch, dim=0)
        # print(num_nodes_per_graph)

        # compute the indices of the self node in each graph
        up_self_index = torch.cumsum(up_num_nodes_per_graph, dim=0) - 1

        up_self_node_embed = up_x[up_self_index]
        up_self_node_embed = torch.sigmoid(up_self_node_embed)

        # for downstream event graph
        down_x, down_edge_index, down_batch = batch_down_data.x.to(torch.float32), batch_down_data.edge_index, batch_down_data.batch
        down_x = self.down_conv1(down_x, down_edge_index)

        # get the number of nodes in each graph in the batch
        down_num_nodes_per_graph = scatter_sum(down_batch.new_ones(
            batch_down_data.num_nodes), down_batch, dim=0)

        # compute the indices of the self node in each graph
        down_self_index = torch.cumsum(down_num_nodes_per_graph, dim=0) - 1

        down_self_node_embed = down_x[down_self_index]
        down_self_node_embed = torch.sigmoid(down_self_node_embed)

        x = up_self_node_embed * down_self_node_embed
        x = self.mlp(x)

        return x

