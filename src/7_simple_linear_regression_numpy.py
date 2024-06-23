import numpy as np


def main():
    """
    [선형 회귀 모델 - 넘파이]
    - 아래 코드는 y = x1 * w + b 와 같은 단순한 선형 회귀 모델을 예측하고, 평가하고, 수정하여 학습하는 기능을,
        토치 라이브러리의 지원을 받지 않고 넘파이로 구현한 것입니다.
        선형 회귀 모델의 기본 형태와 경사 하강법을 이용한 가중치 수정을 통한 머신 러닝 학습 방식을 확인 할 수 있습니다.
    """

    # 학습용 독립 변수 데이터
    train_x = np.array(
        [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
         [11], [12], [13], [14], [15], [16], [17], [18], [19], [20],
         [21], [22], [23], [24], [25], [26], [27], [28], [29], [30]]
    )

    # 학습용 종속 변수 데이터
    train_y = np.array(
        [[0.94], [1.98], [2.88], [3.92], [3.96], [4.55], [5.64], [6.3], [7.44], [9.1],
         [8.46], [9.5], [10.67], [11.16], [14], [11.83], [14.4], [14.25], [16.2], [16.32],
         [17.46], [19.8], [18], [21.34], [22], [22.5], [24.57], [26.04], [21.6], [28.8]]
    )

    # 선형 회귀 모델(y = (x * w) + b) 를 이루는 학습 가능한 파라미터(w, b)
    model_weight = 0.0
    model_bias = 0.0

    # 에포크 (학습 횟수)
    epoch_n = 10000
    # 학습률
    learning_rate = 0.001

    for epoch in range(epoch_n):
        # 모델 예측
        model_out = train_x * model_weight + model_bias

        # 손실 계산 (MSE)
        # 아래에서는 정답값과 예측값의 차이를 구하고, 그것의 부호를 제거하기 위해 제곱을 한 값들의 평균을 구하는 Mean Square Error 를 사용했는데,
        # 선형 회귀가 아닌 이진 분류에서는 Cross Entropy 와 같은 수식을 사용하게 하는 등, 예측모델에 적합한 손실 함수를 사용하면 됩니다.
        # 요는, 머신러닝의 학습의 목표는 이 손실함수의 값을 최소화 하는 것이 됩니다.
        # 비용 함수(cost function) = 손실 함수(loss function) = 오차 함수(error function) = 목적 함수(objective function)
        model_loss = np.mean((model_out - train_y) ** 2)

        # 기울기 계산 (MSE 를 편미분 하는 공식)
        d_weight = 2 * np.mean((model_out - train_y) * train_x)
        d_bias = 2 * np.mean(model_out - train_y)

        # 경사 하강법으로 모델 파라미터 갱신
        model_weight -= learning_rate * d_weight
        model_bias -= learning_rate * d_bias

        if (epoch + 1) % 1000 == 0:
            # 1000 에폭마다 로깅
            print(
                f"Epoch : {epoch + 1:4d}, Weight : {model_weight:.3f}, Bias : {model_bias:.3f}, Cost : {model_loss:.3f}")


if __name__ == '__main__':
    main()
