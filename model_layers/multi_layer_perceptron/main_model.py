from torch import nn


class MainModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 모델 내 레이어
        self.layer1 = nn.Sequential(
            nn.Linear(2, 2),
            nn.Sigmoid()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.layer1[0].weight)
        self.layer1[0].bias.data.fill_(0.01)

        nn.init.xavier_uniform_(self.layer2[0].weight)
        self.layer2[0].bias.data.fill_(0.01)

    def forward(self, model_in):
        model_out = self.layer1(model_in)
        model_out = self.layer2(model_out)
        return model_out
