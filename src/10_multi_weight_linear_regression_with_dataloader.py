import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader


def main():
    """
    [선형 회귀 모델 - 다중 입출력 및 데이터 로더 사용]
    - 이전 예시와 다른 점으론, 선형 회귀 모델이 값 한개를 받아 값 한개를 출력 하는 것이 아니라,
        여러 개의 값을 받아서 여러 개의 값을 출력할 수 있다는 것을 보여 주는 예시 입니다.
        선형 회귀 모델의 수식이 병렬로 여러개 존재 하는 것과 같다고 생각 하면 됩니다.

    - 다차원 데이터를 사용하는 분석/예측 모델은 현실의 인과관계에 빗대어 보자면,
        내일 비가 올지에 대한 예측을, 현재의 기온, 기압, 풍향, 풍속, 습도 등의 여러 데이터를 종합적으로 판단하는 것과 같습니다.
        현실적인 예측 모델에 다차원 데이터를 사용하는 것은 필수이지만,
        무조건 차원이 많다고 좋은 것이 아닙니다.
        앞서 예를 든 내일 비가 오는지에 대한 예측에, 직관적으로도 굉장히 상관도가 낮거나, 혹은 기존의 데이터와 의미가 중복되는 변수를 추가하여 고려한다면,
        해당 데이터에서 오는, 결과와 연관되지 않은 무작위성이 오히려 결과에 악영향을 미치거나, 연산량이 늘어나거나 하는 등의 문제가 생길 수 있습니다.
        차원이 커질수록 나타나는 문제는, '차원의 저주'라는 것이 있으며,
        이는 차원이 늘어날수록 두 데이터 사이의 공간이 커진다는 것이 됩니다. (경우의 수가 많아지고, 특정 데이터가 출현할 확률이 낮아지는 것)
        예를들어 x1, x2 라는 두 축으로 이루어진 2차원에서는 점 두개가 있을 때, x1 축 방향으로 10, x2 축 방향으로 2 만큼 이동하면 두 점이 만난다고 합시다.
        하지만 x1, x2, x3 라는 세 축으로 이루어진 3차원에서는, 점 두개 있다면, x1, x2 방향으로 위와 같이 동일하게 움직여도, x3 축의 좌표가 맞지 않으면 두 점은 만날 수 없죠.
        공간이 늘어난다는 것은 값 하나하나의 영향력이 줄어들고, 그 외에 고려해야할 사항들이 많아진다는 것이기에,
        머신러닝 학습시에, 해당 공간을 설명해줄만한 다양하고 충분한 데이터가 많이 필요하다는 것이 됩니다.
        (한손만 가진 로봇은 손을 들고 내리는 테스트만 하면 된다면, 양손과 양발을 가진 로봇은 한손을 들고,
        양 다리를 접고, 한 손을 내린 테스트, 양 손을 들고 양 다리를 편 테스트... 이렇게 다양한 테스트가 필요한 것과 같습니다.)
        이는 현재 빅데이터를 사용하는 딥러닝 모델의 단점과도 밀접하게 연관된 것으로,
        고로 예측 모델은 결과에 직접적인 영향을 끼치는 중복되지 않는 관측 데이터를 선별하는 것에서 시작됩니다.
        (대부분 모델을 설계하는 사람이 이러한 것들을 결정하게 되는데, 이 역시 자동으로 해주는 알고리즘이 있다면, 머신러닝 개발자를 대체하는 것이 될 것이고,
        현 시점에 LLM 모델에서 이러한 능력을 보이는 모델이 있습니다.)

    - 아래 코드의 모델 학습시 데이터 입력에 필요한 여러 기능(셔플, 데이터 나누기 - 배치, 등...)을 제공 하는
        데이터 로더 객체를 사용할 것 입니다.
    """

    # 학습용 독립 변수 데이터
    train_x = torch.FloatTensor(
        [
            [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]
        ]
    )

    # 학습용 종속 변수 데이터
    train_y = torch.FloatTensor(
        [
            [0.1, 1.5], [1, 2.8], [1.9, 4.1], [2.8, 5.4], [3.7, 6.7], [4.6, 8]
        ]
    )

    # 학습 데이터셋 객체
    train_dataset = TensorDataset(train_x, train_y)

    # 학습 데이터 로더 객체 (배치 사이즈, 셔플 여부, 배치 분할 후 여분 데이터 버릴지 여부)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)

    # 선형 회귀 모델 (2개를 받아서 2개를 반환)
    model = nn.Linear(in_features=2, out_features=2, bias=True)

    # 손실 함수
    criterion = nn.MSELoss()

    # 옵티마이저
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(10000):
        # 에폭별 손실값
        epoch_loss = 0.0

        # 데이터 로더에서 배치 데이터 가져오기
        for batch in train_dataloader:
            x, y = batch

            # 모델 순전파
            model_out = model(x)

            # 비용 함수 계산
            model_loss = criterion(model_out, y)

            # 옵티마이저 초기화
            optimizer.zero_grad()

            # 비용 함수 결과 역전파 = 옵티마이저 기울기 계산
            model_loss.backward()

            # 계산된 기울기로 파라미터(weight, bias) 수정
            optimizer.step()

            epoch_loss += model_loss

        epoch_loss = epoch_loss / len(train_dataloader)

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch : {epoch + 1:4d}, Model : {list(model.parameters())}, Cost : {epoch_loss:.3f}")


if __name__ == '__main__':
    main()
