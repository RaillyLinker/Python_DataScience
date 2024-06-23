from torch import nn


class MainModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 모델 내 레이어
        self.linear = nn.Linear(784, 10, bias=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0.01)

    def forward(self, model_in):
        model_out = self.linear(model_in)
        return model_out
