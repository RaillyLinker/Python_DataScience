import numpy as np

"""
[단일 퍼셉트론]
- 퍼셉트론은, 동물의 신경망을 본뜬 알고리즘으로, 딥러닝의 인공 신경망의 가장 기본적인 형태입니다.
    입력을 받아 출력값을 반환하는 일반적인 선형 회귀 모델과 다른점은,
    신경망은 출력층 마지막에 선형 출력값을 비선형으로 바꿔주는 활성화 함수가 달려있다는 것입니다.
    신경 세포와 마찬가지로 활성화 함수로 출력된 출력값을 다른 퍼셉트론이 받고, 또 그 퍼셉트론의 신호를 다른 퍼셉트론들이 받는 것으로 이어져있는 구조입니다.
    
- 인공 신경망 발전사
    1943년, McCulloch-Walter Pitts 의 논문에 신경을 그물망 형태로 연결하면 사람의 뇌처럼 동작할 수 있다는 가능성을 주장합니다.
        이전에는 인공적으로 사람의 정신과 지능을 구현하기 위해 무엇이 필요한지에 대해 방향도 잡지 못하던 상황에,
        사람의 뇌 신경, 뇌 구조 자체를 복제한다는 방향성을 잡아준 것으로 의의가 있죠.
    1957년, 미국의 신경 생물학자 Frank Rosenblatt 가 위 개념을 실제 장치로 만들어 선보였으며,
        이것이 바로 로젠블라트 퍼셉트론으로, 입력값과 정답을 통해 선형 회귀 모델의 가중치를 학습 시킬 수 있는 최초의 모델입니다.
    1960년, Bernard Widrow와 Tedd Hoff 가 로젠블라트의 한계를 극복하기 위해 Adeline 퍼셉트론을 개발하였으며, 학습시 경사 하강법을 도입하였습니다.
        아델라인은 학습 가능한 선형 회귀 모델로써 그 성능을 입증한 이후 서포트 벡터 머신 등 머신러닝의 중요한 알고리즘들로 발전합니다.
    1969년, 아델라인이 개발된 후 인공지능 개발이 실현되리라 기대되었지만, 
        인공지능 분야의 선구자였던 MIT 의 Marvin Minsky 교수의 Perceptrons 라는 논문에, 
        퍼셉트론은 Xor 문제를 해결할 수 없다는 큰 결점이 있다는 것을 증명하였고, 
        당시에는 이것을 해결하는 동시에 스스로 학습하는 구조를 유지하는 방법을 찾을 수 없었습니다.
        
- Xor 문제
    XOR 는 논리 회로로,
    0 0 -> 0
    0 1 -> 1
    1 0 -> 1
    1 1 -> 0
    위와 같은 진리표에서 보이듯, 동일한 값이 입력되면 0, 서로 다른 값이 입력되면 1 을 반환하는 회로입니다.
    아델라인 퍼셉트론은 AND, OR, NOT 의 회로는 구현이 가능해도, XOR 은 구현할 수 없습니다.
    이는, 퍼셉트론이 가장 근본적인 논리구조의 하나를 해결하지 못한다는 뜻이 됩니다.
"""
# (단층 퍼셉트론으로 논리 회로 구현)
# AND 게이트를 위한 입력 및 출력 데이터
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_and = np.array([[0], [0], [0], [1]])

# OR 게이트를 위한 입력 및 출력 데이터
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_or = np.array([[0], [1], [1], [1]])

# XOR 게이트를 위한 입력 및 출력 데이터
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_xor = np.array([[0], [1], [1], [0]])

# 가중치(weight) 및 편향(bias) 초기화
W = np.random.rand(2, 1)
b = np.random.rand(1)

# 학습률(learning rate)
lr = 0.01


# 활성화 함수 (임계값을 0으로 설정하여 step function으로 만듦)
def step_function(x):
    return 1 if x > 0.5 else 0


# 손실 함수 (binary cross entropy)
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# 정확도 계산 함수
def accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)


# 퍼셉트론 학습
def perceptron_learning(X, Y):
    global W, b
    losses = []
    accuracies = []
    for i in range(len(X)):
        x = X[i]
        y = Y[i]

        # 예측
        z = np.dot(x, W) + b
        output = step_function(z)

        # 오차 계산
        error = y - output
        loss = binary_cross_entropy(y, output)
        losses.append(loss)

        # 정확도 계산
        acc = accuracy(y, output)
        accuracies.append(acc)

        # 가중치 업데이트
        W += lr * error * x.reshape(-1, 1)

        # 편향 업데이트
        b += lr * error

    avg_loss = np.mean(losses)
    avg_acc = np.mean(accuracies)
    return avg_loss, avg_acc


# AND 게이트 학습
print("AND 게이트 학습 시작")
for i in range(100):
    avg_loss, avg_acc = perceptron_learning(X_and, Y_and)
    if (i + 1) % 10 == 0:
        print(f"Epoch {i + 1} - Loss: {avg_loss}, Accuracy: {avg_acc}")

print("\nAND 게이트 학습 완료")
print("가중치:", W)
print("편향:", b)

# OR 게이트 학습
print("\nOR 게이트 학습 시작")
for i in range(100):
    avg_loss, avg_acc = perceptron_learning(X_or, Y_or)
    if (i + 1) % 10 == 0:
        print(f"Epoch {i + 1} - Loss: {avg_loss}, Accuracy: {avg_acc}")

print("\nOR 게이트 학습 완료")
print("가중치:", W)
print("편향:", b)

# XOR 게이트 학습
print("\nXOR 게이트 학습 시작")
for i in range(100):
    avg_loss, avg_acc = perceptron_learning(X_xor, Y_xor)
    if (i + 1) % 10 == 0:
        print(f"Epoch {i + 1} - Loss: {avg_loss}, Accuracy: {avg_acc}")

print("\nXOR 게이트 학습 완료")
print("가중치:", W)
print("편향:", b)

"""
- 위의 코드에서 퍼셉트론의 구조가 잘 나타나 있습니다.
    실제 생명체의 신경 구조와 동일하게, 입력값을 받아들이는 부분이 있고, 
    이를 선형 회귀 모델로 증폭 혹은 감소 시키며,
    역치(신경에서 일정 자극 이하면 해당 신호를 무시하고, 이상이라면 자극을 전달)를 구현한 활성화 함수인 계단 함수(Step Function) 를 통과하여, 
    최종 결과를 출력하는 구조입니다.
    특히, 일반적인 선형 회귀 모델과 퍼셉트론이 다른 점으로는 활성화 함수의 유무입니다.
    활성화 함수는 선형 함수 뒤에 붙으면, 함수의 선형적인 결과값을 비선형적으로 변환해주는 효과가 있습니다.
    학습시에는 경사 하강법을 사용한 것도 확인하세요.
    
- 활성화 함수는 꼭 위와 같은 계단 함수가 아니어도 됩니다.
    후에 설명할 Sigmoid, ReLU, Tanh 등의 함수가 있으며,
    중요한 것은, 활성화 함수는 비선형 함수여야 한다는 것입니다.
    만약 활성화 함수로 선형 함수를 사용한다면, 이어지는 앞뒤 선형함수가 분리되지 않고,
    그저 하나의 선형 함수로 합쳐지기 때문입니다.
    
- 이제 위 코드의 실행 결과를 확인해겠습니다. 
    위와 같은 단일 퍼셉트론으로는 AND 와 OR 회로는 Accuracy 1.0 으로 잘 학습이 되지만, 
    XOR 은 절대 1.0 이 될 수 없음을 볼 수 있습니다.
    이유는, XOR 의 경우는 선형회귀 모델의 선 하나로 절대 분류가 불가능한 형태를 띄고 있기 때문입니다.
    
- 단순히 생각하여, 이를 해결하기 위해서는 선을 2개를 사용해야 합니다.
    더 복잡한 문제라면 선을 더 사용해야겠죠.
    이러한 컨셉에서 나온 XOR 의 해결 방식이 바로 다층 퍼셉트론 구조이며, 이것을 학습시키기 위해 나온 것이 오차역전파입니다.
    신경망의 구조와, 그 구조를 학습시키기 위한 방법론...
    두 개념 모두 현재 딥러닝의 핵심이 됩니다.
    
- 추가로, XOR 을 회로로 봅시다.
    AND, OR, NOT 과 같은 회로는 그대로 만들 수 있지만, XOR 게이트의 경우는 다른 회로와 결합이 필요합니다.
    AND 게이트에 NOT 을 취한 NAND 게이트의 결과값과, OR 게이트의 결과값을 AND 게이트에 통과하면 XOR 게이트가 됩니다.
    [1, 1] 만 허용하는 AND 게이트에 NOT 을 취하여 [1, 0], [0, 1], [0, 0] 일 때를 허용하는 NAND 게이트가 되고,
    [1, 0], [0, 1], [1, 1] 일 때를 허용하는 OR 게이트와 AND 게이트를 적용시키면,
    두 결과값 중 공통되는 [1, 0], [0, 1] 만 허용하게 되므로 XOR 이 구현된 것이죠.
"""


# (다층 퍼셉트론 구조로 XOR 회로를 해결)
# 퍼셉트론
def MLP(x, w, b):
    y = np.sum(w * x) + b
    if y <= 0.5:
        return 0
    else:
        return 1


# 신경망 게이트
def NAND(x1, x2):
    return MLP(np.array([x1, x2]), np.array([-2, -2]), 3)


def OR(x1, x2):
    return MLP(np.array([x1, x2]), np.array([2, 2]), -1)


def AND(x1, x2):
    return MLP(np.array([x1, x2]), np.array([1, 1]), -1)


# XOR 게이트는 AND 게이트 내부에 NAND 와 OR 게이트가 결합되어 있습니다.
# 이는 내부의 게이트들이 계산한 결과를 받아와서 외부의 게이트가 처리하므로, 다층 구조를 지닙니다.
def XOR(x1, x2):
    return AND(NAND(x1, x2), OR(x1, x2))


# 진리표상의 입력값을 입력하여 출력값이 잘 나오는지를 확인
print("\n다층 퍼셉트론으로 XOR 구현 결과")
for x in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    y = XOR(x[0], x[1])
    print("입력 값 : " + str(x) + " 출력 값 : " + str(y))

"""
- 위 코드에서 다층 퍼셉트론이 XOR 문제를 잘 해결함을 확인할 수 있습니다.
    
- 구조를 봅시다.
    앞서 단층 퍼셉트론에서도 퍼셉트론에 붙는 활성화 함수가 선형 모델의 결과의 선형성을 끊어주는 역할을 한다고 했는데,
    이렇게 선형성이 끊긴 단층 퍼셉트론이 서로 연결되니, 각 논리 게이트별 독립된 역할을 결합하여 사용할 수 있게 된 것입니다.
    이렇게 되면 단층 퍼셉트론을 여러개 붙여 사용한다는 것은,
    기계에서 마치 다양한 논리 게이트를 붙여서 커다란 논리 회로를 만들 듯이 설계를 하는 것과 같다는 것이 됩니다.
    정리하자면 기본 논리 회로를 구현하기 위한 퍼셉트론의 구조는,
    입력층과 출력층 사이에 은닉층이 하나 필요하다는 것이 됩니다.
    
- 이처럼 구조는 만들었지만, 당시는 입력값을 이용하여 스스로 학습하게 하는 방법은 없었습니다.
    단일 퍼셉트론은 입력값과 출력값만으로 이루어지고, 그 안에 파라미터가 한 층으로 이루어져 있으므로 손실함수 값을 기준으로 편미분을 한번만 하면 되지만,
    다층 퍼셉트론의 경우를 봅시다. 
    출력층에 이어진 파라미터 층을 손실함수 값을 기준으로 편미분을 하는 것으로 같지만, 
    은닉층의 파라미터는 무엇을 기준으로 미분을 해야할지 알 수 없기에 은닉층을 학습시킬 수 없는 것이며,
    즉, 이 시점의 다층 퍼셉트론은 그저 논리회로 수동으로 구현한 수식인 것이지, 머신러닝이 아니었던 것입니다.
    이를 해결하기 위해서는, 그로부터 약 20여 년간의 시간이 필요했으며, 이 기간을 인공지능의 겨울이라고 불렀습니다.

- 인공지능의 겨울 기간,
    퍼셉트론에서 이어나가던 인공지능 개발의 방법론은 두가지로 나뉩니다.
    하나는, 생명의 신경망을 모방하여 완전한 인공지능을 완성하는 것은 놔두고, 
    최적화된 예측선을 잘 그려주던 머신러닝 회귀모델인 아달라인을 발전시켜, 
    SVM 이나 로지스틱 회귀 모델 등의 보다 빠르게 개발 및 사용이 가능한 머신러닝 모델을 개발한 그룹과,
    다른 하나는 계속해서 다층 퍼셉트론의 학습 방법을 찾던 그룹입니다.
    Geoffrey Hinton 교수는 1986 년, 오차 역전파 방식을 발명해내어 인공지능의 겨울을 극복하였으며,
    딥러닝의 아버지라 불리게 됩니다.
    
    이후에는 다층 퍼셉트론을 중심으로, 인공신경망 모델이 발전을 거듭해나가다가,
    CPU 연산 능력, GPU 를 통한 병렬 연산 등의 하드웨어적인 발전을 만나서 어느순간 기대한 성능을 내기 시작하였고,
    알파고를 통한 인간 지능에의 정면 대결 등으로 큰 주목을 받았으며,
    ChatGPT, 미디어 생성 AI 등의 눈부신 성과가 현재 일반인들의 눈 앞에도 보일 수준이 되었습니다.
    
- 자, 이쯤하여 인공지능의 겨울을 극복한 역전파에 대해 알아보겠습니다.

- 순전파 (Forward Propagation)
    인공 신경망에서 순전파란, 모델에 입력값을 넣어 예측을 하는 작업을 말합니다.
    ax + b 라는 모델로 구성된 신경망에, 2 라는 x 값이 입력되면, a2 + b 라는 값이 나오는 것입니다.
    다층 퍼셉트론이 앞선 레이어의 계산 결과를 다음 레이어로 전파해나가며 최종 결과를 반환하는 것이므로,
    이처럼 순방향으로 계산을 하는 것이 순전파라고 합니다.
    역전파에 대해 알기 전에 순전파에 대해 알아보았는데,
    역전파는 순전파와는 다르게, 계산이 모델의 끝부분에서 앞쪽으로 진행된다는 것을 쉽게 추론할 수 있습니다.
    
- 역전파 (Backpropagation)
    오차 역전파의 정체는 편미분과 체인룰로 설명이 가능합니다.
    체인룰이란, 합성함수에서 최종적인 미분값은 각 함수별 국소 미분값의 곱이라는 법칙입니다.
    
    앞서, 다층 퍼셉트론의 은닉층 파라미터의 편미분을 무엇을 기준으로 구하는지 알 수 없기에 학습이 불가능하다고 했는데,
    결국 모델 전체로 보자면 손실함수의 값을 줄이는 것이 목표인 것은 같습니다.
    즉, 은닉층의 목표 역시 손실함수의 값을 줄이는 것이며, 이를 기준으로 편미분을 하면 된다는 것입니다.
    
    이렇게 된다면, 다층 신경망 모델은 하나의 합성함수이며, 은닉층을 포함한 레이어별 파라미터의 편미분값 역시 체인룰로 구할 수 있습니다.
    
    합성함수인 다층 퍼셉트론의 끝에서부터 봅시다.
    가장 끝에 위치하는 w3 의 손실값에 대한 편미분 값을 f`(w3) 라고 하면,
    그 앞에 위치하여, 순전파시 w3 에게 입력값을 전달한 파라미터인 w2 의 편미분은,
    f`(w2) * f`(w3) 가 됩니다.
    즉, w1 -> w2 -> w3 로 계산이 진행되던 순전파와는 다르게,
    편미분 값을 구하려면, w3 -> w2 -> w1 으로 손실 함수의 값이 전파되며 기울기가 계산되는 것입니다.
    
    이러한 계산 방식으로 인하여 순전파와 정반대로 진행되는 역전파... 오차를 역전파한다고 하여 오차 역전파라고 불리는 것입니다.
    
    간단한 예시를 통해 봅시다.
    
    1. 조건 :
        w = 3
        x = 2
        b = 1
        모델 = y = w * x + b
        정답 = yc = 4
        학습률 = 0.01
    2. 순전파 : 3 * 2 + 1 = 7
    3. 손실 계산 : (4 - 7)^2 = 4
    4. 기울기 계산 : 오차 역전파를 통해 기울기(gradient)를 계산하는 과정은 손실 함수의 가중치에 대한 미분을 통해 이루어집니다.
        손실 함수 출력값에 영향을 끼치는 파라미터인 w 와 b 의 각 미분을 구하면,
        wd = (2 * (y-yc) * x) = (2 * (7-4)*2) = 12
        bd = (2 * (y - yc)) = (2 * (7 - 4)) = 6
    5. 가중치 업데이트 : 기울기가 계산되었으니, 이제 이를 사용해 가중치 w와 편향 b를 업데이트할 수 있습니다.
        w = w - 학습률 * wd = (3 - 0.01 * 12) = 2.88
        b = b - 학습률 * bd = (1 - 0.01 * 6) = 0.94
        
    위와 같이 가중치를 업데이트 할 수 있습니다.
    
    Torch 에서 제공해주는 Tensor 를 사용하여 계산을 진행하면, 위와 같은 역전파에 대한 로직을 구현할 것 없이 앞서도 사용해보았던
    
    loss.backward()
    
    라는 함수로 위와 같이 각 파라미터로 역전파하여 기울기를 구할 수 있는 것입니다.
"""

"""
[경사 하강법 종류]
- 경사 하강법은 파라미터가 손실함수에 끼치는 영향력을 편미분으로 구하여 일정 학습률로 조금씩 수정하는 것을 반복하여,
    손실 함수의 값이 최소가 되도록 하는 파라미터 수정 방법입니다.
    이 방식은 최적 해를 찾기까지 계산량과 시간이 많이 걸린다는 단점이 존재하는데,
    이를 개선하기 위한 여러 방법들이 개발되었습니다.
    딥러닝에서는 학습시 적용되는 경사 하강법을 옵티마이저(Optimizer)라고 부릅니다.
    딥러닝 프레임워크에서 제공하는 Optimizer 클래스로 이를 쉽게 적용할 수 있습니다.
    아래에서 여러 옵티마이저들의 종류와 내용을 알아봅시다.
    
(확률적 경사 하강법(Stochastic Gradient Descent, SGD))
- 일반 경사 하강법(Gradient Descent, GD) 를 개선한 첫번째 경사 하강법입니다.

(모멘텀(Momentum))
- 확률적 경사 하강법에서 발전하여, 관성의 방향을 고려해 진동과 폭을 줄이는 효과를 내도록 하였습니다.

(아다그라드(Adagrad))
- 확률적 경사 하강법에서 발전하여, 변수의 업데이트가 잦으면 학습률을 적게 하여 이동 보폭을 조절하였습니다.

(네스테로프 모멘텀(NAG))
- 모멘텀에서 발전하여, 모멘텀이 이동시킬 방향으로 미리 이동하여 불필요한 이동을 줄이는 효과를 내도록 하였습니다.

(알엠에스프롭(RMSProp))
- 아다그라드에서 발전하여, 보폭 민감도를 보완하였습니다.

(아담(Adam))
- 모멘텀과 알엠에스프롭을 합친 방법입니다.
"""