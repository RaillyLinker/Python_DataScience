from torch import nn


class MainModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 모델 내 레이어
        self.layer = nn.Sequential(
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.layer[0].weight)
        self.layer[0].bias.data.fill_(0.01)

    def forward(self, model_in):
        model_out = self.layer(model_in)
        return model_out
