from torch import nn
from torch.nn import functional as F

from src.types_ import *


class GateNet(nn.Module):
    def __init__(self, feature_num, candidate_num, hidden_dims: List = None, relu: bool = True):
        super(GateNet, self).__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]
        self.relu = relu
        self.feature_num = feature_num
        self.candidate_num = candidate_num
        self.hidden_dims = hidden_dims
        self.weights_num = 0
        current_dim = self.feature_num
        for hidden_dim in self.hidden_dims:
            self.weights_num += current_dim * hidden_dim + hidden_dim
            current_dim = hidden_dim
        self.weights_num += current_dim * self.candidate_num + self.candidate_num

    def forward(self, X, weight_params) -> NpArray:
        assert len(weight_params) == self.weights_num
        current_index = 0
        current_dim = self.feature_num
        current_value = X
        for index, hidden_dim in enumerate(self.hidden_dims):
            weight = weight_params[current_index:current_index + current_dim * hidden_dim].view(hidden_dim, current_dim)
            current_index += current_dim * hidden_dim
            bias = weight_params[current_index:current_index + hidden_dim]
            current_index += hidden_dim
            current_value = F.linear(current_value, weight, bias)
            if self.relu:
                current_value = F.relu(current_value)
            current_dim = hidden_dim
        weight = weight_params[current_index:current_index + current_dim * self.candidate_num].view(self.candidate_num,
                                                                                                    current_dim)
        current_index = current_index + current_dim * self.candidate_num
        bias = weight_params[current_index:current_index + self.candidate_num]
        current_index += self.candidate_num
        final_value = F.linear(current_value, weight, bias)
        return final_value.cpu().detach().numpy()


if __name__ == '__main__':
    gate_net = GateNet(feature_num=81, candidate_num=27, hidden_dims=[128, 128])
    weight = torch.DoubleTensor(np.array(np.random.random(gate_net.weights_num), dtype=np.float64))
    y = gate_net(torch.DoubleTensor(np.array(np.random.random((256, 81)), dtype=np.float64)), weight)
    print()
