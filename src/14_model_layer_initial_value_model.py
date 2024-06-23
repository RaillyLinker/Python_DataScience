import torch
from torch import nn

"""
[모델 파라미터 초기화]
- 아래 코드는 토치에서 제공하는 NN 레이어의 파라미터를 수동으로 설정할 수 있다는 것을 보여줍니다.
    주로 적절한 초기값을 설정하여 모델 학습을 보조하는 방식으로 사용 합니다.
"""


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(2, 2),
            nn.Sigmoid()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

        # 모델 초기값 설정
        self.layer1[0].weight.data = torch.nn.Parameter(
            torch.Tensor([[0.4352, 0.3545],
                          [0.1951, 0.4835]])
        )

        self.layer1[0].bias.data = torch.nn.Parameter(
            torch.Tensor([-0.1419, 0.0439])
        )

        self.layer2[0].weight.data = torch.nn.Parameter(
            torch.Tensor([[-0.1725, 0.1129]])
        )

        self.layer2[0].bias.data = torch.nn.Parameter(
            torch.Tensor([-0.3043])
        )
