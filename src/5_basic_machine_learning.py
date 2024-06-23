import numpy as np

"""
[머신러닝 기본]
- 머신러닝은 학습하는 알고리즘입니다.
    함수가 데이터를 입력하면 미리 정해진 알고리즘으로 결과를 도출해내는 것이라면,
    머신러닝은 데이터를 입력받아 올바른 결과를 도출하는 함수를 만들어내는 것을 의미합니다.
    함수가 그저 도구라면, 머신러닝은 도구를 만드는 도구라고 할 수 있겠네요.

- 머신러닝은 데이터 분석의 도구이자, 인공지능을 달성하는 방법 중 하나입니다.
    데이터를 다루고, 데이터를 이해한다는 것은, 관측되는 데이터의 일부를 사용하여 전체, 혹은 일부 데이터를 예측할 수 있다는 것이 되므로,
    인공지능을 만들든, 데이터를 분석하든 머신러닝에선 예측 모델을 이해하고 만드는 것이 중요합니다.

- 예측 모델에 대해 설명하겠습니다.
    예측 모델은 크게 2가지 유형이 있습니다.
    회귀와 분류...
    회귀란, 다음에 올 값을 맞추는 모델입니다.
    예를들어, 빵값이 1월에는 100원, 2월에는 120원, 3월에는 130원 이라는 데이터가 있을 때, 다음월인 4월에는 얼마가 될지를 예측하는 것입니다.
    분류란, 값이 어디에 속하는지를 맞추는 모델입니다.
    10, 0 라는 값들의 묶음이 있을 때, 이것을 A 라고 두고, 4, 9 이라는 묶음은 B 라고 할 때, 1, 8 이라는 값은 어디에 속할지에 대해 맞추는 것입니다.
    즉, 이 두개의 예측 모델을 이해하고, 이를 자동으로 만들어낼 수 있는 방법론을 찾는 것이 머신러닝의 목표입니다.

(선형회귀)
- 예측 모델의 첫번째이자 딥러닝을 이루는 가장 기본적인 개념입니다.
    회귀 문제 중 선형 회귀란, 예측 모델의 결과값이 입력값의 선형적인 변화에 따라 선형적으로 변화되는 것을 의미합니다.
    y = ax + b 라는 일차 함수로 나타낼 수 있으며, x가 선형적으로 변경되면, y 역시 선형적으로 변경되는 것을 볼 수 있죠.
    용어 정리를 먼저 하겠습니다.
    y 는 종속 변수로, 함수 우변의 값에 종속된 값입니다.
    예측의 결과값으로, 그때 그때 주어지는 데이터에 종속되어 값이 변경되므로 종속변수이죠.
    x 는 독립변수입니다.
    집값을 예측할 때, 예측의 목표가 되는 집값이 y 값인 종속변수이면, 현실에서 주어지는 데이터, 예를들면 집값에 영향을 주는 월간 범죄 건수를 x 라고 하면,
    이는 현실에 존재하며, y에 영향을 끼치는 독립된 값이므로 독립변수지요.
    a는 가중치입니다.
    x 가 y 에 영향을 끼치는 정도를 의미하며, x 가 증가할 때, a 의 값이 클수록 y 에 끼치는 영향이 커집니다.
    b 는 편향으로, a 가 클수록 x 가 y 에 끼치는 영향력이 증가한다고 하면, b 는 x 가 0이어도 b이고, x 가 무한정 커져도 b 인 값입니다.

- 앞서 예시를 든, 독립 변수의 종류가 1개 뿐인 단순 선형 회귀뿐 아니라, 독립 변수의 종류가 여러개인 다중 선형회귀도 존재합니다.
    y = a1x1 + a2x2 + a3x3 + b
    이렇게 만들 수도 있는데,
    이는 해석하자면, x1, x2, x3 라는 독립 변수가 y 에 끼치는 영향을 수식화 한 것입니다.
    가중치인 a 역시 여러개이므로, 각 독립변수의 영향력이 서로 다름을 나타냅니다.

- 선형회귀 모델을 머신러닝으로 구현해본다고 합시다.
    이때의 목표는, 주어진 독립변수 x 와 y 를 놔두고, 모델을 이루는 파라미터인 a 와 b 를 구하는 것입니다.
    이를 자동으로 구하기 위해 가장 먼저, 단순 선형 회귀 모델에 대한 파라미터를 찾는 최소 제곱법이라는 방식을 알아보겠습니다.

- 최소 제곱법 (Least Squares Method)
    입력값 x 에 대한 관측값이자 정답값인 y 를 도출할 수 있도록 일차함수의 파라미터를 수정하는 방법에는 최소 제곱법이 있습니다.

    최소 제곱법은,
    "실제값과 모델이 도출한 예측값 사이의 차이인 잔차의 제곱을 최소화 하는 것"을 목표로 하기에 최소 제곱법이라고 불립니다.
    이때, 잔차에 제곱을 하는 이유는 부호를 제거하기 위한 목적과, 차이가 커질수록 값이 커진다는 성질을 사용하기 위한 것입니다.

    일차함수에서 파라미터를 구하는 수식은 아래와 같습니다.

    a = sum((x - {x 의 평균}) * (y - {y 의 평균})) / sum((x - {x 의 평균})^2)
    b = {y 의 평균} - ({x 의 평균} * {위에서 구한 a})

    a 를 구하는 수식을 알아보겠습니다.

    1. 일차 함수는, f(x) = ax + b 의 형태를 가집니다.

    2. 잔차의 제곱합(SSE)을 최소화 하는 것이 최소 제곱법의 목표입니다.
        e = y - f(x)
        잔차의 계산은 위와 같고,
        x 와 y 값이 여러개 존재하므로,
        sum(e^2) = sum((y - f(x))^2)
        이렇게 제곱합을 계산합니다.

    3. 잔차의 제곱합 공식을 a 에 대한 편미분을 하여 그 값이 0 이 되도록 하면 됩니다.
        SSE 의 편미분이 0 이 되도록 하는 것은, 잔차의 제곱합을 0 으로 만드는 것이 아닙니다.

        미분의 값이 0 이라는 것은, 해당 위치에서 변수가 되는 파라미터 a 나 b 가 약간 변동이 있더라도,
        결과값에 영향을 끼치지 않는다는 것으로,
        SSE 는 아래로 오목한 토기 형태를 띕니다.

        고로 기울기가 0 이 된다는 것은, SSE 가 최소가 되는 것을 의미하기에,
        SSE 가 0 이 되는 것이 아니라, SSE 가 나타낼수 있는 최소 값을 찾았다는 의미가 되는 것이죠.

        SSE 공식은,
        SSE = sum((y - ax - b)^2)

        여기서 a 로 편미분을 하기 위하여 체인룰을 적용합시다.
        체인룰은, 전체 미분의 값은 내부 함수 미분 값을 모두 곱한 것과 같다는 것으로,

        가장 외곽의 e^2 을 먼저 미분하면,
        미분법칙에 의하여,

        2(y -ax - b)

        여기에 내부 함수 y -ax - b 를 미분하면, y 와 b 가 사라지고, -x 가 남습니다.

        즉, -2x(y - ax - b)

        에서, 편미분시 b 는 상수 취급되므로

        -2x(y - ax)

        가 되며, 즉

        -2 * sum(x(y - ax)) = 0

        이 되도록 하면 됩니다.

        이것 역시 풀어보겠습니다.

        앞의 -2 는 상수이므로 결과에 영항을 끼치지 못하므로,

        sum(x(y - ax)) = 0

        을 하면 됩니다.

        = sum(xy - a(x)^2)
        = (sum(xy)) - a(sum(x^2))

        즉, 0 = (sum(xy)) - a(sum(x^2))

        이것을 a 를 구하는 식으로 변경하기 위해 좌우항으로 분배하면,

        a(sum(x^2)) = (sum(xy))

        마지막으로 이것을 a 를 구하는 공식으로 정리하면,

        a = sum(xy) / sum(x^2)

        이렇게 전개됩니다.

    파라미터 b 를 구하는 것에 대해서는 역시나 SSE 를 b 에 대해 편미분하여 그 값이 0 이 되도록 b 를 구하면 됩니다.

    - 앞서 설명한 잔차의 제곱합 공식을 b 에 대한 편미분을 하여 그 값이 0 이 되도록 하면 됩니다.
        SSE 공식은,
        SSE = sum((y - ax - b)^2)

        여기서 a 로 편미분을 하기 위하여 체인룰을 적용합시다.
        체인룰은, 전체 미분의 값은 내부 함수 미분 값을 모두 곱한 것과 같다는 것으로,

        가장 외곽의 e^2 을 먼저 미분하면,
        미분법칙에 의하여,

        2(y -ax - b)

        여기에 내부 함수 y -ax - b 를 미분하면, y 와 -ax 가 사라지고, -1 이 남습니다.

        즉, -2(y - ax - b)

        -2 * sum(y - ax - b) = 0

        이 되도록 하면 됩니다.

        이것 역시 풀어보겠습니다.

        앞의 -2 는 상수이므로 결과에 영항을 끼치지 못하므로,

        sum(y - ax) - b = 0
        = sum(y - ax) - b

        b = sum(y - ax)

        이렇게 전개됩니다.

    - 수식을 살펴보았으니 실제로 일차함수 파라미터를 구하는 예시를 알아보겠습니다.

"""
# (최소 제곱법)
# 아래 예시 데이터에서 x 는 공부 시간, y 는 수학 성적을 의미합니다.
# 두 데이터 간 선형적인 연관관계를 구해보겠습니다.
x = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97])

# 각 평균값
mx = np.mean(x)
my = np.mean(y)

# 기울기 공식 분모
divisor = sum([(i - mx) ** 2 for i in x])


# 기울기 공식 분자
def top(x, mx, y, my):
    d = 0
    for i in range(len(x)):
        d += (x[i] - mx) * (y[i] - my)
    return d


dividend = top(x, mx, y, my)

# 기울기 계산
a = dividend / divisor
b = my - (mx * a)

# 구한 기울기를 사용하여 선형 회귀
fx = a * x + b
print(f"numpy a : {a}, numpy b : {b}")
print(f"numpy y : {y}")
print(f"numpy fx : {fx}")

"""
[평균 제곱 오차 (Mean Square Error : MSE)]
- 위에서 언급한 최소 제곱법의 단점이 존재합니다.
    1. 비선형적인 문제, 복잡한 모델에 적용하기에 적합하지 못하다.
    2. 대량의 데이터에서 계산 비용이 많아진다.
    입니다.
    
    1번의 경우는, 최소 제곱법이 현재 존재하는 모든 데이터를 기반으로 하여 현 상황의 최적의 기울기를 구하는 것이지만,
    노이즈에도 최적화 될 가능성이 있으므로 복잡한 모델일수록 과적합 되기가 쉽다는 의미이고,
    
    2번의 경우는, 모델의 모든 파라미터에 대하여 계산을 행해야 하는데, 파라미터 개수 * 데이터 개수 * 데이터 특징 개수 로,
    복잡하고 많은 데이터의 최적 모델을 찾는데 어마어마한 시간과 연산이 들어가게 됩니다.
    
    즉, 빅데이터 + 복잡한 모델에서 최소 제곱법을 사용할 수 없다는 것입니다.
    
    모델의 파라미터를 찾는 다른 머신러닝 알고리즘으로는 경사 하강법이 존재하는데,
    여기서 간단히 설명 하자면, 모델이 결정되었으면, 그 모델의 파라미터를 임의로 설정하고,
    그 모델의 오차를 나타내는 오차 함수를 준비한 후,
    모델을 실행 -> 오차 함수로 평가 -> 오차 함수 값이 낮아지는 방향으로 현 상황의 각 파라미터의 편미분 값을 구하기 -> 
    각 파라미터들의 편미분 값의 방향 및 크기를 기반으로 파라미터 약간 수정 -> 다시 모델을 실행 -> 평가 및 수정 로직 반복...
    이렇게 반복을 하는 것입니다. (쉽게 말하면, 에러가 작아지는 방향으로 조금씩 파라미터를 수정하기)
    
    자세한 것은 뒤에서 정리합니다.
    
- 평균 제곱 오차는, 경사 하강법에서 사용하는 기본적인 오차 함수입니다.
    잔차의 제곱합 (SSE) 에서 데이터 개수만큼 나눠주면 MSE 가 됩니다.
    즉,
    MSE = sum((실제 값 - 예측값) ^2) / n
"""


# (평균 제곱 오차 구현)
# MSE 함수
def mse(y, y_pred):
    return (1 / len(y)) * sum((y - y_pred) ** 2)


# 데이터
x = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97])

# 임의로 정한 파라미터
fake_a = 3
fake_b = 76

# 선형 회귀 모델 계산
pred_y = fake_a * x + fake_b

# MSE 값 출력
print(mse(y, pred_y))

"""
[경사 하강법]
- 위의 오차 함수의 값을 봅시다.
    머신러닝의 목적은 위와 같은 오차 함수를 최소로 만드는 것이고, 그것을 위해 모델의 파라미터를 스스로 수정합니다.
    MSE 오차 함수 값의 특징은, 잔차에 제곱을 취했으므로, 정답에 가까울 수록 경사가 완만해지고, 정답에 멀어질수록 경사가 가팔라집니다.
    정답값이 10이라고 합시다.
    예측값이 동일하게 10이라면, 잔차의 제곱은 0일 것입니다.
    예측값이 9 라면, 잔차의 제곱은 1 이겠네요.
    예측값이 8이라면 잔차의 제곱은 4 가 되며, 7이라면 9, 6이라면 16... 이렇게 차이가 커지면 값도 커집니다.
    예측값이 작아도 그러하며, 11, 12, 14... 이렇게 예측값이 커지면서 오차가 발생해도 동일하기에,
    오차함수 값의 그래프는 정답 부분이 가장 낮은, 아래로 오목한 밥그릇 형태입니다.
    
- 경사하강법은, 위와 같이 정답일수록 값이 작은 오차 함수의 특성에 따라, 현재 오차의 경사면을 타고 오차 그래프의 가장 낮은 위치로 탐색해나가는 
    머신러닝의 파라미터 갱신 알고리즘입니다.
    말은 간단한데, 실제 컴퓨터로 이를 구현하려면 어떻게 해야할까요?
    탐색 알고리즘을 생각하면 되는데,
    현 상황에서 어느 방향으로 이동할지, 얼마나 이동할지를 먼저 확인해야 합니다.
    이는 미분을 통하여 알 수 있습니다.
    파라미터의 변화에 따른 현 상황, 오차함수 결과값의 기울기를 알 수 있다면, 그 부호로 파라미터의 변경 방향을 알 수 있고,
    그 수치의 크기로 파라미터의 변경 크기를 가늠할 수 있습니다.

- 이미 위에서 설명을 했지만, 경사 하강법에 대하여 다시 로직을 정리해보자면,
    모델을 실행 -> 오차 함수로 평가 -> 오차 함수 값이 낮아지는 방향으로 현 상황의 각 파라미터의 편미분 값을 구하기 -> 
    각 파라미터들의 편미분 값의 방향 및 크기를 기반으로 파라미터 약간 수정 -> 다시 모델을 실행 -> 평가 및 수정 로직 반복...
    이렇게 반복을 하는 것입니다. (쉽게 말하면, 에러가 작아지는 방향으로 조금씩 파라미터를 수정하기)
    
    위와 같습니다.
    여기서, 
    오차 함수 미분을 한 시점에, 탐색해야하는 최소 오차가 어디에 위치한지는 모르는 상황입니다.
    즉, 얼마만큼 파라미터를 수정해야하는지는 정확히 알 수 없는 상황으로,
    얼마나 이동을 할 지에 대한 설정값을 학습률이라고 합니다.
    
    학습률 * 기울기
    
    를 하면, 파라미터를 수정하는 값이 나오게 되는데, 기울기에 따라서, 오차가 커질수록 파라미터 변경 폭이 커지게 되며,
    학습률이 커지거나 작아짐에 따라, 한번에 수정되는 파라미터의 변동량이 달라지게 됩니다.
    
    학습률이 너무 크다면, 에러의 최저점을 지나치게 될 수가 있고,
    학습률이 너무 작다면, 학습 횟수가 그만큼 많아지고, 추후 설명할 것인데, 
    복잡한 모델의 경우는 국소 부위에 기울기가 0 이 되는 지점이 있을 수 있기에, 전체에서 최저점을 찾기 전에 이곳에서 기울기가 0 이라고 판단하여 
    학습을 멈추게 되는 기울기 소실 문제가 발생할 수도 있습니다.
"""
