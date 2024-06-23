import torch
from torch import optim

"""
[선형 회귀 모델 - 토치]
- 아래 코드는 머신러닝의 기본적인 두가지 문제인 회귀 문제, 분류 문제 중 회귀 문제에 속하는 가장 단순한 선형 회귀 모델을
    토치 라이브러리의 지원을 받아 구현한 것입니다.
"""

# 학습용 독립 변수 데이터
train_x = torch.FloatTensor(
    [
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
        [11], [12], [13], [14], [15], [16], [17], [18], [19], [20],
        [21], [22], [23], [24], [25], [26], [27], [28], [29], [30]
    ]
)

# 학습용 종속 변수 데이터
train_y = torch.FloatTensor(
    [
        [0.94], [1.98], [2.88], [3.92], [3.96], [4.55], [5.64], [6.3], [7.44], [9.1],
        [8.46], [9.5], [10.67], [11.16], [14], [11.83], [14.4], [14.25], [16.2], [16.32],
        [17.46], [19.8], [18], [21.34], [22], [22.5], [24.57], [26.04], [21.6], [28.8]
    ]
)

# 선형 회귀 모델(y = (x * w) + b) 를 이루는 학습 가능한 파라미터(w, b)
model_weight = torch.zeros(1, requires_grad=True)
model_bias = torch.zeros(1, requires_grad=True)

# 옵티마이저
optimizer = optim.SGD([model_weight, model_bias], lr=0.001)

for epoch in range(10000):
    # 선형 회귀 모델 순전파
    model_out = train_x * model_weight + model_bias

    # 비용 함수 (MSE) 실행
    model_loss = torch.mean((model_out - train_y) ** 2)

    # 옵티마이저 기울기 초기화
    # 파이토치는 미분을 통해 얻은 기울기를 이전에 계산된 기울기 값에 누적시키는 특징이 있습니다.
    # 나중에 가면 이 값이 축적되며 기울기가 가팔라져 가기만 하는데,
    # 이를 초기화하여 이번 학습에 필요한 기울기만을 사용하기 위해서 초기화를 해야합니다.
    optimizer.zero_grad()

    # 비용 함수 결과 역전파 = 옵티마이저 기울기 계산
    # requires_grad 가 True 로 설정된 텐서의 계산이 model_loss 에 영향을 끼치는 기울기를 자동으로 계산해줍니다.
    # backward() 함수 뒤에 model_weight.grad 이렇게 조회하여 기울기를 확인 가능합니다.
    model_loss.backward()

    # 계산된 기울기로 파라미터(weight, bias) 수정
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        # 1000 에폭마다 로깅
        print(f"Epoch : {epoch + 1:4d}, "
              f"Weight : {model_weight.item():.3f}, "
              f"Bias : {model_bias.item():.3f}, "
              f"Cost : {model_loss:.3f}")
