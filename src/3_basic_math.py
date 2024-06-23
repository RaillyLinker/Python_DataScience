import matplotlib.pyplot as plt
import numpy as np

"""
[딥러닝 기본 수학]
- 여기선 딥러닝에 사용되는 가장 기본적인 수학 및, 본 예시 리스트를 이해하는데 있어 필요한 수학적 지식에 대해 정리 하겠습니다.
    또한 이를 파이썬 코드로 구현하였으므로, 파이썬 코드에서는 각 계산을 어떻게 작성하여 사용하는지를 훑어보고 갑시다.

(일차함수)
- 일차함수의 가장 기본적인 형태는,
    y = ax + b
    로 볼 수 있습니다.
    입력값 x 가 있을 때, 이에 a 를 곱하고 b 를 더한 것으로,
    a 는 기울기라고 하고, b 는 절편이라고 합니다.
    먼저 기울기란, 입력값 x 가 변화할 때, y 가 변화하는 방향(기호)과 크기(숫자의 크기)를 말하며,
    절편은, x 가 0 일 때에도 기본적으로 y 가 가지고 있는 위치를 의미합니다.
    다음으로, 일차 함수의 일차라는 것은 위 수식에서 a 의 지수를 의미합니다.
    지수란, a^2, a^3 와 같이 제곱, 세제곱을 나타내는 것에서, a^2 에서는 2, a^3 에서는 3 과 같이 위에 달리는 숫자로,
    아시다시피 a 를 몇번 곱하느냐에 대한 값인데, 함수 내에 가장 지수가 큰 기울기의 지수가 1 이라면 일차함수, 2 라면 이차함수라고 합니다.
    
- 일차함수는 그래프로 나타내면 선이 하나 그어지는 것으로,
    기울기인 a 의 경우는, x 가 증가 할 때 y 가 변화하는 정도로, 예를들어 기울기가 1 이라면 x 의 증가값과 동일하게 증가하고,
    2 라면 y 는 x 의 2배씩 증가하게 될 것이며, 0이라면 y 는 x 값에 전혀 영향을 받지 못하게 됩니다.
    절편인 b 의 경우는 x 가 0 일 때의 y 의 초기값으로 생각하면 됩니다.
"""


def linear_function_sample():
    # x 값 생성 (-10부터 10까지 0.1 간격으로)
    x = np.arange(-10, 10, 0.1)
    # 일차함수 식
    y = 2 * x + 3
    # 그래프 그리기
    plt.plot(x, y, label=f'2x + 3')
    plt.title('Linear Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()

    # (이차함수의 그래프)
    # 이차함수 그래프를 본다면 아래(기울기가 0보다 클 때)로 불룩한 토기 모양입니다.
    # x 값 생성 (-10부터 10까지 0.1 간격으로)
    x = np.arange(-10, 10, 0.1)
    # 일차함수 식
    y = x ** 2
    # 그래프 그리기
    plt.plot(x, y, label=f'x ** 2')
    plt.title('Quadratic Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()


"""
(미분)
- 위와 같은 이차 함수의 그래프는 곡선을 그리고 있습니다.
    x 가 1에서 2로 갈 때의 y의 변화 정도와, x 가 4에서 5로 갈 때의 y의 변화 정도는 다를 것입니다.
    그 간격을 더 작게 하여, 1에서 1.5 로 갈 때, 1 에서 1.00000001 로 갈 때의 y 값의 변화 정도는 또 다를 것입니다.
    이렇게 작게 쪼갠다고 하여 미분입니다.
    이러한 방식으로 x라는 점에서의 기울기를 알아낼 수 있게 됩니다.
    
- 미분 기울기를 구하는 공식
    (y 값의 증가량 / x 값의 증가량) = ((f(b) - f(a)) / (b - a)) = ((f(a + dx) - f(a)) / (dx)) = (dy / dx)
    위 공식에서 dx 는 x 값의 극소 증가값입니다. 
    dy 는 dx 가 적용 되었을 때의 y 의 증가 값이죠.
    dx 로 dy 를 나눈다는 것은, dx 의 값이 dy 에 끼치는 영향력을 나타낸 것입니다.
    예를 들어 봅시다. x 가 1 일때의 기울기를 구하고 싶다면,
    함수에 넣을 수 있는 매우 작은 값을 0.0001 이라고 결정했다고 합시다.
    그러면 x 가 1.0001 일때의 y 값과 x 가 1 일때의 y 값의 차이를 구하면 dy 가 준비되겠죠?
    dx 의 경우는 1.0001 에서 1 을 빼면 됩니다.
    
- 미분 법칙
    1. 함수 f(x) = x 일 때, f(x) 의 미분 함수인 f`(x) = 1 입니다. (함수에서 입력값과 동일한 값이 반환되니 기울기는 1)
    2. f(x) = a 에서 a 가 상수일 때, f`(x) = 0 (함수의 입력값이 어떻든 동일한 상수값이 반환되므로 기울기가 없습니다.)
    3. f(x) = ax 에서 a 가 상수일 때, f`(x) = a (입력값에 a 가 곱해져 나오는 형태이므로, a 가 즉 기울기가 됩니다.)
    4. f(x) = x^a 에서 a 가 자연수일 경우, f`(x) = ax^(a-1)
    5. 함성 함수 f(g(x)) 에서 f(x) 와 g(x) 가 미분 가능할 때 {f(g(x))}` = f`(g(x)) * g`(x) 
        (합성 함수의 기울기는 각 함수의 미분값의 곱과 같습니다. 이를 체인룰이라고 부릅니다.)
    
    위와 같은 법칙을 사용하여 보다 복잡한 미분을 보다 빠르게 계산이 가능합니다.
    
- 편미분
    함수 내에 변수가 여러개 있을 때, 특정 변수만의 기울기를 구하는 것,
    나머지 변수는 특정 값으로 고정하여 상수로 보고 미분을 하는 것을 편미분이라 부릅니다.
    예를들어 f(x1, x2) = x1^2 + (x1 * x2) + a 라는 함수가 있을 때, x1 에 대해 편미분을 한다고 하면,
    df / dx1 이라고 표시를 하며, 미분 법칙에 따라 정리하면,
    df / dx1 = 2x1 + x2 가 됩니다.
    4번 법칙으로 2x1 이 나온 것이고, 
    2번 법칙으로 x2 가 편미분으로 상수취급이 되므로 x2 가 나온 것이고, 
    a 의 경우는 상수이므로 2번 법칙으로 0 이 된 것입니다.
    
- 체인룰
    체인룰은, 전체 미분의 값은 내부 함수들의 미분 값을 모두 곱한 것과 같다는 것으로, 추후 정리할 딥러닝의 역전파 알고리즘에 중요한 역할을 합니다.
    예를들어 f(x) 가 내부적으로 여러 함수들의 결과로 이루어진 합성함수라고 해봅시다.
    f(f(x) + f(f(x) * f(x)))
    이렇게, 내부 함수의 결과로 계산을 한 결과가 다음 계산에 사용되는 형식입니다.
    이렇게 된다면 이 함수의 미분값은,
    가장 안쪽의 내부함수의 미분 값 * 외곽 미분 값 * .... 이렇게 되는 것입니다.
"""


def differential_sample():
    # (미분 결과 그래프)
    # 특정 미분 위치에서 미분을 수행한 결과를 그래프로 나타냅니다.
    # 미분 위치
    differentiation_x = 7

    # 이차 함수 그래프
    def quadratic_function(x):
        return x ** 2

    # 미분 함수
    def derivative_quadratic(x):
        return 2 * x

    # x 값 생성 (-10부터 10까지 0.1 간격으로)
    x_positive = np.arange(0, 10, 0.1)
    # 이차함수 값 계산 (음수 부분 제외)
    y_positive = quadratic_function(x_positive)
    # 미분값 계산 (x=7에서의 미분값)
    derivative_at_x7 = derivative_quadratic(differentiation_x)
    # 미분값을 이용하여 접선 계산
    tangent_line_at_x7_positive = derivative_at_x7 * (x_positive - differentiation_x) + quadratic_function(
        differentiation_x)
    # 그래프 그리기
    plt.plot(x_positive, y_positive, label=f'x ** 2 (positive)')
    plt.plot(x_positive, tangent_line_at_x7_positive, label=f'Tangent at x=7', linestyle='--', color='red')
    plt.scatter(differentiation_x, quadratic_function(differentiation_x), color='red')  # 해당 점 표시
    plt.title('Quadratic Function and Tangent at x=7 (positive)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()


"""
(지수와 지수함수)
- 지수는 앞서 설명했듯, 숫자의 제곱 횟수입니다.
    1제곱이라면 지수는 1, 제곱이라면 2, 세제곱이라면 3 입니다.
    
- 지수의 아래에 있는 제곱되는 대상의 숫자는 밑이라고 부릅니다.

- 지수함수란, 변수 x 가 지수 자리에 있는 함수를 의미합니다.
    y = a^x
    와 같습니다.
    이러한 지수함수에서는 밑의 값이 무엇인지가 중요한데, 이 값이 1 이라면 함수가 아니며, 0 보다 작으면 허수를 포함하게 되므로 안됩니다.
    
- 아래는 밑의 값이 1 보다 클 경우와 0보다 크고 1 보다 작을 때의 경우를 나타낸 지수함수의 그래프입니다.
"""


def exponential_sample():
    # x 값 생성
    x = np.linspace(-2, 2, 400)
    # a > 1 일 때의 지수함수 그래프 (a=2)
    y1 = 2 ** x
    # 0 < a < 1 일 때의 지수함수 그래프 (a=0.5)
    y2 = 0.5 ** x
    # 그래프 그리기
    plt.plot(x, y1, label=r'$y=2^x$ (a > 1)')
    plt.plot(x, y2, label=r'$y=0.5^x$ (0 < a < 1)')
    # 그래프 스타일 및 레이블 설정
    plt.title('Exponential Functions with Different Values of a')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()


"""
[시그모이드 함수]
- 시그모이드 함수는 x 를 받아서 0 에서 1 사이의 값을 반환합니다.
    입력값인 x 가 작은 값을 가지면 f(x) 값은 한없이 0에 가까워지는 값을 반환하고,
    입력값인 x 가 큰 값을 가지면 f(x) 값은 한없이 1에 가까워지는 값을 반환합니다.
    
- 수식 : 
    1 / (1 + np.exp(-x))
    
- 이는 머신러닝에서 x 를 0 에서 1 사이의 실수값으로 변환하는 데이터 정제에 사용하는 함수로 사용됩니다.
"""


def sigmoid_sample():
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # -7부터 7까지 0.1 간격으로 x 값을 생성
    x_values = np.arange(-7, 7, 0.1)

    # 시그모이드 함수 계산
    y_values = sigmoid(x_values)

    # 그래프 그리기
    plt.plot(x_values, y_values, label='Sigmoid Function')
    plt.title('Sigmoid Function')
    plt.xlabel('x')
    plt.ylabel('sigmoid(x)')
    plt.legend()
    plt.grid(True)
    plt.show()


"""
[로그와 로그 함수]
- a ** x = b 라는 수식이 있다고 합시다.
    지수 함수는 밑인 a 와 지수인 x 가 존재할 때, b 의 갑을 구하는 문제인데,
    반대로 a 와 b 가 존재할 때, 지수인 x 를 구하고 싶을 수가 있습니다.
    밑과 결과값이 존재할 때, 지수를 구하는 법칙이 바로 로그입니다.
    
    영어로 Logarithm 이라고 하며, 줄여서 log 라고 표시됩니다.

- 위의 a ** x = b 지수 함수를 참고하여 로그 함수를 수식으로 나타내면,
    log_a(b) = x 라고 할 수 있습니다.

- 파이썬에서는 math.log(b, a) 혹은 np.log(b) / np.log(a) 로 계산합니다.

- x 가 1 에서 0 사이의 확률값이라 합시다.
    -log(x) 는 x 값이 커질수록 값이 작아집니다. 
    -log(1-x) 와 같이 입력하면, x 값이 커질수록 값이 커지죠.
    추후 이것을 통해 로지스틱 회귀와 같은 확률 모델을 학습 시킬 때 사용하는 Cross Entropy 손실 함수를 설명 할 수 있으므로,
    아래 코드의 그래프를 확인하세요.
"""


def log_sample():
    # 로그 함수의 정의
    def logarithmic_function(x, base):
        return np.log(x) / np.log(base)

    def neg_logarithmic_function(x, base):
        return -np.log(x) / np.log(base)

    def neg_logarithmic_function_rev(x, base):
        return -np.log(1 - x) / np.log(base)

    # 그래프의 x 범위 설정
    x_values = np.linspace(0.01, 1, 100)

    # 베이스 값 설정 (양의 실수, 1이 아닌 값)
    base_value = 2.718

    # 함수 그래프 그리기
    plt.figure(figsize=(18, 6))

    # 로그 함수 그래프
    plt.subplot(1, 3, 1)
    plt.plot(x_values, logarithmic_function(x_values, base_value), label=f'Logarithmic (base={base_value})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    # -log 함수 그래프
    plt.subplot(1, 3, 2)
    plt.plot(x_values, neg_logarithmic_function(x_values, base_value),
             label=f'Negative Logarithmic (base={base_value})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    # -log 함수 그래프
    plt.subplot(1, 3, 3)
    plt.plot(x_values, neg_logarithmic_function_rev(x_values, base_value),
             label=f'Negative Logarithmic Rev (base={base_value})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    # linear_function_sample()
    # differential_sample()
    # exponential_sample()
    # sigmoid_sample()
    log_sample()


if __name__ == '__main__':
    main()
