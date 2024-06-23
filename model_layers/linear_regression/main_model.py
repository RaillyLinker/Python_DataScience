from torch import nn


class MainModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 모델 내 레이어
        self.linearLayer = nn.Linear(2, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linearLayer.weight)
        self.linearLayer.bias.data.fill_(0.01)

    def forward(self, model_in):
        model_out = self.linearLayer(model_in)
        return model_out
