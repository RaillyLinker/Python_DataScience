import torch
from torch import nn
from torch import optim

"""
[선형 회귀 모델 - 토치 레이어]
- 이 샘플이 이전의 토치 라이브러리를 사용한 선형 회귀 모델 샘플과 다른 점으론, 
    파라미터가 모델 레이어 객체 안에 포함되어 제공된다는 것이고,
    손실 함수도 토치 라이브러리에서 제공된 것을 사용하게 됩니다.
    앞으로 딥러닝 개발은 이처럼 레이어 단위로 예측 모델을 쌓아가며 만드는 것이 주가 될 것입니다.
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

# 선형 회귀 모델
model = nn.Linear(in_features=1, out_features=1, bias=True)

# 손실 함수
criterion = nn.MSELoss()

# 옵티마이저
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10000):
    # 모델 순전파
    model_out = model(train_x)

    # 비용 함수 계산
    model_loss = criterion(model_out, train_y)

    # 옵티마이저 초기화
    optimizer.zero_grad()

    # 비용 함수 결과 역전파 = 옵티마이저 기울기 계산
    model_loss.backward()

    # 계산된 기울기로 파라미터(weight, bias) 수정
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch : {epoch + 1:4d}, "
              f"Model : {list(model.parameters())}, "
              f"Cost : {model_loss:.3f}")
