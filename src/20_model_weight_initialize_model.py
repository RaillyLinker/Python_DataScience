from torch import nn

"""
[모델에서 가중치를 초기화 하는 방법]
- 가중치를 초기화 한 모델 사용 방법입니다.
    앞으로 모델을 작성할 때에는 아래와 같이 초기화를 설정하는 것이 좋습니다.

- 세이비어 초기화(Xavier Initialization)
    2010년 세이비어 글로럿과 요슈아 벤지오는 가중치 초기화가 모델에 미치는 영향을 분석하여 새로운 초기화 방법을 제안했습니다. 
    이 초기화 방법은 제안한 사람의 이름을 따서 세이비어(Xavier Initialization) 초기화 또는 글로럿 초기화(Glorot Initialization)라고 합니다.
    이 방법은 균등 분포(Uniform Distribution) 또는 정규 분포(Normal distribution)로 초기화 할 때 두 가지 경우로 나뉘며, 
    이전 층의 뉴런 개수와 다음 층의 뉴런 개수를 가지고 식을 세웁니다. 
    이전 층의 뉴런의 개수를 in, 다음 층의 뉴런의 개수를 out 이라고 해봅시다.
    글로럿과 벤지오의 논문에서는 균등 분포를 사용하여 가중치를 초기화할 경우,
    -sqrt(6/in+out) ~ sqrt(6/in+out) 사이의 균등 분포 범위를 사용하라고 합니다.
    정규 분포로 초기화할 경우에는 평균이 0이고, 표준 편차 σ가 sqrt(2/in + out) 을 만족하도록 합니다.
    세이비어 초기화는 여러 층의 기울기 분산 사이에 균형을 맞춰서 특정 층이 너무 주목을 받거나 다른 층이 뒤쳐지는 것을 막습니다. 
    그런데 세이비어 초기화는 시그모이드 함수나 하이퍼볼릭 탄젠트 함수와 같은 S자 형태인 활성화 함수와 함께 사용할 경우에는 좋은 성능을 보이지만, 
    ReLU와 함께 사용할 경우에는 성능이 좋지 않습니다. 
    ReLU 함수 또는 ReLU의 변형 함수들을 활성화 함수로 사용할 경우에는 다른 초기화 방법을 사용하는 것이 좋은데, 
    이를 He 초기화(He initialization)라고 합니다.
    
- He 초기화(He initialization)
    He 초기화(He initialization)는 세이비어 초기화와 유사하게 정규 분포와 균등 분포 두 가지 경우로 나뉩니다. 
    다만, He 초기화는 세이비어 초기화와 다르게 다음 층의 뉴런의 수를 반영하지 않습니다. 전과 같이 이전 층의 뉴런의 개수를 in 이라고 해봅시다.

    He 초기화는 균등 분포로 초기화 할 경우에는 다음과 같은 균등 분포 범위를 가지도록 합니다.
    -sqrt(6/in) ~ sqrt(6/in)
    정규 분포로 초기화할 경우에는 표준 편차 σ가 sqrt(2/in) 을 만족하도록 합니다.

- 시그모이드 함수나 하이퍼볼릭탄젠트 함수를 사용할 경우에는 세이비어 초기화 방법이 효율적입니다.

- ReLU 계열 함수를 사용할 경우에는 He 초기화 방법이 효율적입니다.
"""


class MainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(20, 10),
            nn.Sigmoid()
        )
        self.layer3 = nn.Linear(in_features=10, out_features=1)

        self._init_weights()

    def _init_weights(self):
        # ReLU 를 활성화 함수로 사용하는 layer1 에는 Kaiming / He 초기화를 적용
        nn.init.kaiming_uniform_(self.layer1[0].weight, mode='fan_in', nonlinearity='relu')
        self.layer1[0].bias.data.fill_(0)

        # 그 외에는 Xavier 초기화를 적용
        nn.init.xavier_uniform_(self.layer2[0].weight)
        self.layer2[0].bias.data.fill_(0.01)

        nn.init.xavier_uniform_(self.layer3.weight)
        self.layer3.bias.data.fill_(0.01)

        # RNN, LSTM, GRU 등의 재귀적 모델에는 직교 초기화 (Orthogonal Initialization 을 사용하세요.)

    def forward(self, model_in):
        model_out = self.layer1(model_in)
        model_out = self.layer2(model_out)
        model_out = self.layer3(model_out)
        return model_out
