import torch
from torch import nn
from torch.utils.data import DataLoader
import utils.torch_util as tu
import model_layers.multi_layer_perceptron.main_model as multi_layer_perceptron
import os
from torch import optim

"""
[다층 퍼셉트론]
- 단층 퍼셉트론을 여러개 쌓은 형태의 모델입니다.
    단층 퍼셉트론을 선 하나로 생각하세요. (선 하나를 그어두고 그 위에 있는 것은 1, 아래에 있는 것은 0으로 분류)
    단층 퍼셉트론 끝에 달린 활성화 함수로 모델의 선형성을 끊었는데,
    이렇게 끊어진 선들이 결합됨으로써 선형적으로는 판별할 수 없는 여러 문제들을 해결 할 수 있습니다. (Xor 문제 등)
    한붓 그리기에서 일반적인 그리기로 업그레이드 된 것입니다.
    퍼셉트론은 데이터와 정답이 있다면 내부적으로 자동으로 학습을 하기에 이 선들이 많다면 세상에 풀지 못할 문제가 없습니다.
    이것이 기본 딥러닝 모델의 원리입니다. 
- 다층 퍼셉트론에서 데이터를 처음 받는 부분을 입력층이라 하며, 신경망 내부에서 연산한 결과를 출력하는 곳을 출력층이라고 합니다.
    그 사이의 중첩이 가능한 중간 부분을 은닉층이라 부르는데, 이 은닉층이 쌓이면 쌓일수록 복잡한 문제를 해결 가능하기에,
    복잡한 문제를 푸는 인공 신경망 모델은 허리가 길어지죠.
    은닉층이 많다는 것은 Deep 하다는 것이겠죠? 그리고 그러한 Deep 한 은닉층들을 학습시키는 머신러닝이라고 하여 이름을 딥러닝이라 부르는 것입니다.
- 
"""


def main():
    # 사용 가능 디바이스
    device = tu.get_gpu_support_device(gpu_support=True)

    # 데이터셋 객체 생성 (ex : tensor([[-10., 100., 82.], ...], device = cpu), tensor([[327.7900], ...], device = cpu))
    # CSV 파일로 데이터셋 형성 (1 행에 라벨이 존재하고, 그 라벨로 x, y 데이터를 분류 합니다.)
    dataset = tu.CsvModelDataset(
        csv_file_full_url="../resources/datasets/17_multi_layer_perceptron_nn/perceptron.csv",
        x_column_labels=['x1', 'x2'],
        y_column_labels=['y1']
    )

    # 학습용, 검증용, 테스트용 데이터를 비율에 따라 분리
    train_dataset, validation_dataset = tu.split_dataset(
        dataset=dataset,
        train_data_rate=0.8,
        validation_data_rate=0.2
    )

    # 데이터 로더 래핑
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=10, shuffle=True, drop_last=True)

    # 모델 생성
    model = multi_layer_perceptron.MainModel()

    # 모델 학습
    tu.train_model(
        device=device,
        model=model,
        criterion=nn.BCELoss(),
        optimizer=optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.001),
        train_dataloader=train_dataloader,
        num_epochs=10000,
        validation_dataloader=validation_dataloader,
        check_point_file_save_directory_path="../_check_point_files/multi_layer_perceptron",
        # check_point_load_file_full_path="../_check_point_files/~/checkpoint(2024_02_29_17_51_09_330).pt",
        log_freq=1000
    )

    # 모델 저장
    model_file_save_directory_path = "../_torch_model_files/multi_layer_perceptron"
    if not os.path.exists(model_file_save_directory_path):
        os.makedirs(model_file_save_directory_path)
    save_file_full_path = tu.save_model_file(
        model=model,
        model_file_save_directory_path=model_file_save_directory_path
    )

    # # 저장된 모델 불러오기
    model = torch.load(save_file_full_path, map_location=device)
    print("Model Load Complete!")
    print(model)

    # 모델 순전파
    with torch.no_grad():
        model.eval()
        inputs = torch.FloatTensor([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]).to(device)
        outputs = model(inputs)

        print("---------")
        print(outputs)
        print(outputs <= 0.5)


if __name__ == '__main__':
    main()

"""
[시그모이드 함수(Sigmoid function)와 기울기 소실]
- 시그모이드 함수의 문제점은 미분을 해서 기울기(gradient)를 구할 때 발생합니다.
    
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    
    plt.plot(x, y)
    plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
    plt.title('Sigmoid Function')
    plt.show()
    
    위 코드는 시그모이드 함수 그래프를 그리는 코드입니다.
    이것으로 출력되는 그래프를 시그모이드 함수의 출력값이 0 또는 1에 가까워지면, 그래프의 기울기가 완만해지는 모습을 볼 수 있습니다.
    
    완만해지는 부분은 기울기를 계산하면 0에 가까운 아주 작은 값이 나오게 됩니다. 
    그런데 역전파 과정에서 0에 가까운 아주 작은 기울기가 곱해지게 되면, 앞단에는 기울기가 잘 전달되지 않게 됩니다. 
    이러한 현상을 기울기 소실(Vanishing Gradient) 문제라고 합니다.
    기울기 소실은 은닉층의 층이 깊을수록 더 심각하게 나타납니다.
    
    시그모이드 함수를 사용하는 은닉층의 개수가 다수가 될 경우에는 0에 가까운 기울기가 계속 곱해지면 앞단에서는 거의 기울기를 전파받을 수 없게 됩니다. 
    다시 말해 매개변수가 업데이트 되지 않아 학습이 되지를 않습니다.
    
    딥러닝은 인공 신경망이 겹겹이 쌓여서 깊은 은닉층을 가지는 것을 의미합니다.
    은닉층이 깊은 신경망에서 기울기 소실 문제로 인해 출력층과 가까운 은닉층에서는 기울기가 잘 전파되지만, 
    앞단으로 갈수록 기울기가 제대로 전파되지 않는 모습을 보여줍니다.
    
    시그모이드 함수의 또 다른 문제점은 원점 중심이 아니라는 점입니다(Not zero-centered). 
    따라서, 평균이 0이 아니라 0.5이며, 
    시그모이드 함수는 항상 양수를 출력하기 때문에 출력의 가중치 합이 입력의 가중치 합보다 커질 가능성이 높습니다. 
    이것을 편향 이동(bias shift)이라 하며, 
    이러한 이유로 각 레이어를 지날 때마다 분산이 계속 커져 가장 높은 레이어에서는 활성화 함수의 출력이 0이나 1로 수렴하게 되어 
    기울기 소실 문제가 일어날 수 있습니다.
    
    결론적으로 시그모이드 함수를 은닉층에서 사용하는 것은 지양됩니다.

[활성화 함수 소개]
- 하이퍼볼릭탄젠트 함수(Hyperbolic tangent function)
    하이퍼볼릭탄젠트 함수(tanh)는 입력값을 -1과 1사이의 값으로 변환합니다.
    아래는 함수를 시각화 한 것입니다.
    
    x = np.arange(-5.0, 5.0, 0.1) # -5.0부터 5.0까지 0.1 간격 생성
    y = np.tanh(x)
    
    plt.plot(x, y)
    plt.plot([0,0],[1.0,-1.0], ':')
    plt.axhline(y=0, color='orange', linestyle='--')
    plt.title('Tanh Function')
    plt.show()
    
    하이퍼볼릭탄젠트 함수도 형태는 시그모이드와 같습니다. 
    극으로 갈수록 기울기가 완만해지죠.
    다만, 0에서 1 사이의 값을 출력하는 것이 아닌, -1과 1에 가까운 출력값을 출력합니다.
    시그모이드 함수와는 달리 0을 중심으로 하고 있는데, 
    이때문에 편향 이동이 일어나지 않으며, 시그모이드 함수와 비교하면 반환값의 변화폭이 더 큽니다. 
    그래서 시그모이드 함수보다는 기울기 소실 증상이 적은 편입니다. 
    그래서 은닉층에서 시그모이드 함수보다는 많이 사용됩니다.

- 렐루 함수(ReLU)
    인공 신경망에서 가장 최고의 인기를 얻고 있는 함수입니다. 
    오차 역전파 방식을 발명하였으며, 딥러닝의 아버지라 불리는 Geoffrey Hinton 교수가 만들었으며,
    수식은 f(x) = max(0, x) 로 아주 간단합니다.
    
    def relu(x):
    return np.maximum(0, x)

    x = np.arange(-5.0, 5.0, 0.1)
    y = relu(x)
    
    plt.plot(x, y)
    plt.plot([0,0],[5.0,0.0], ':')
    plt.title('Relu Function')
    plt.show()
    
    렐루 함수는 음수를 입력하면 0을 출력하고, 양수를 입력하면 입력값을 그대로 반환합니다. 
    렐루 함수는 특정 양수값에 수렴하지 않으므로 깊은 신경망에서 시그모이드 함수보다 훨씬 더 잘 작동합니다. 
    뿐만 아니라, 렐루 함수는 시그모이드 함수와 하이퍼볼릭탄젠트 함수와 같이 어떤 연산이 필요한 것이 아니라 단순 임계값이므로 연산 속도도 빠릅니다.

    하지만 여전히 문제점이 존재하는데, 입력값이 음수면 기울기가 0이 됩니다. 
    입력값의 경우는 값을 조정하는 것으로 개선이 가능한데,
    어떻게든 기울기가 0 이 된 뉴런을 다시 회생하는 것이 매우 어렵습니다.
    이 문제를 죽은 렐루(dying ReLU)라고 합니다.
    
- 리키 렐루(Leaky ReLU)
    죽은 렐루를 보완하기 위해 ReLU의 변형 함수들이 등장하기 시작했습니다. 
    변형 함수는 여러 개가 있지만 여기서는 Leaky ReLU에 대해서만 소개합니다. 
    Leaky ReLU는 입력값이 음수일 경우에 0이 아니라 0.001과 같은 매우 작은 수를 반환하도록 되어있습니다.

    수식은 렐루와 비슷하게, f(x) = max(ax, x) 로 아주 간단합니다. 
    a는 하이퍼파라미터로 Leaky('새는') 정도를 결정하며 일반적으로는 0.01의 값을 가집니다. 
    여기서 말하는 '새는 정도'라는 것은 입력값의 음수일 때의 기울기를 비유하고 있습니다.
    
    a = 0.1
    
    def leaky_relu(x):
    return np.maximum(a*x, x)

    x = np.arange(-5.0, 5.0, 0.1)
    y = leaky_relu(x)
    
    plt.plot(x, y)
    plt.plot([0,0],[5.0,0.0], ':')
    plt.title('Leaky ReLU Function')
    plt.show()
    
    위의 그래프 코드에서는 새는 모습을 확실히 보여주기 위해 a를 0.1로 잡았습니다. 
    입력값이 음수라도 기울기가 0이 되지 않으면 ReLU는 죽지 않습니다.
    
- 소프트플러스(SoftPlus)
    소프트 플러스 함수 역시 렐루 함수를 개선했다고 보면 됩니다.
    렐루 함수는 음수를 입력하면 무조건 0을 출력하고, 양수를 입력하면 입력값을 그대로 반환하는데,
    소프트 플러스 함수는, 0이 되는 기준을 완화시켜서, 
    0으로 시작되는 부분은 마치 시그모이드 함수의 시작처럼 서서히 가파르게 올라가며,
    이후에는 렐루 함수와 같이 한도가 없이 상승하게 됩니다.
    f(x) = log(1+e^x)
    입니다.
    
- 소프트맥스 함수(Softamx function)
    은닉층에서 ReLU(또는 ReLU 변형) 함수들을 사용하는 것이 일반적이지만 
    그렇다고 해서 앞서 배운 시그모이드 함수나 소프트맥스 함수가 사용되지 않는다는 의미는 아닙니다. 
    분류 문제를 로지스틱 회귀와 소프트맥스 회귀를 출력층에 적용하여 사용합니다.
    
    x = np.arange(-5.0, 5.0, 0.1) # -5.0부터 5.0까지 0.1 간격 생성
    y = np.exp(x) / np.sum(np.exp(x))
    
    plt.plot(x, y)
    plt.title('Softmax Function')
    plt.show()
    
    소프트맥스 함수는 시그모이드 함수처럼 출력층의 뉴런에서 주로 사용되는데, 
    시그모이드 함수가 두 가지 선택지 중 하나를 고르는 이진 분류 (Binary Classification) 문제에 사용된다면 
    세 가지 이상의 (상호 배타적인) 선택지 중 하나를 고르는 다중 클래스 분류(MultiClass Classification) 문제에 주로 사용됩니다.

- 은닉층에서는 ReLU나 Leaky ReLU와 같은 ReLU 함수의 변형들을 사용하세요.

- 스탠포드 대학교의 딥 러닝 강의 cs231n에서는 ReLU를 먼저 시도해보고, 
    그다음으로 LeakyReLU나 ELU 같은 ReLU의 변형들을 시도해보며, 
    sigmoid는 사용하지 말라고 권장합니다.
    
[오차 함수 선정법]
- 오차함수는 출력층에 따라 선정하면 됩니다.
    이진분류 문제에는 Sigmoid 와 같은 활성화 함수를 사용하게 되는데, 이는 BCELoss 를 사용하고,
    다중 클래스 분류 문제에는 Softmax 와 같은 활성화 함수를 사용하며, CrossEntropyLoss 를 사용합니다.
    회귀 문제에는 활성화 함수를 붙이지 않으며 MSE 를 사용합니다.
    간단히는 위와 같이 알고 있으면 대부분의 경우 적용이 가능합니다.
"""
