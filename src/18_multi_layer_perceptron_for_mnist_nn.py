import torch
from torch import nn
from torch.utils.data import DataLoader
import utils.torch_util as tu
import model_layers.multi_layer_perceptron_for_mnist.main_model as multi_layer_perceptron_for_mnist
import os
from torch import optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms

"""
[다층 퍼셉트론으로 MNIST 분류하기]
- 앞서 소프트맥스 회귀로 MNIST 데이터를 분류하는 실습을 해봤습니다. 
    소프트맥스 회귀 또한 인공 신경망이라고 볼 수 있는데, 
    입력층과 출력층만 존재하므로 소프트맥스 함수를 활성화 함수로 사용한 '단층 퍼셉트론'이라고 할 수 있습니다. 
    이번 챕터에서는 은닉층을 추가로 넣어 다층 퍼셉트론을 구현하고, 딥 러닝을 통해서 MNIST 데이터를 분류해봅시다.
    
- 아래 코드에서 기존의 MINST 회귀 코드와 다른 것은, 
    기존에는 레이어가 출력 레이어 단 하나뿐이었지만, 지금은 출력 레이어와 합쳐 총 3개의 레이어로 이루어진 모델을 사용한다는 것 뿐입니다.
    실제로 두 모델의 학습 결과를 확인하면,
    기존 모델의 loss 가 0.273 정도라면, 이번 모델의 loss 는 0.076 정도로 매우 낮아진 것을 볼 수 있습니다.
    또한 상황에 따라 다르겠지만, training loss 와 validate loss 간의 차이가 커진 것도 볼 수 있을 것입니다.
    즉, xor 을 포함하여 더 복잡한 문제를 풀 수 있는 딥 뉴럴 네트워크 모델의 성능과 그 학습시 과적합(Overfitting)의 문제를 여실히 확인 할 수 있습니다.
- 저번 코드와의 비교 결과로 인하여 다층 신경망 모델에서 학습시 loss 와 검증시 loss 간의 차이가 벌어질 수 있다는 것을 보았습니다.
    머신러닝 모델은 학습을 많이 하는 것이 중요하죠. 그렇기에 빅데이터의 중요성이 부각되는 것입니다.
    그러나 동일한 데이터를 사용하고 또 사용한다면, 해당 데이터에만 특화된 모델이 만들어지는 것입니다.
    해당 데이터에만 존재하는 노이즈나 특징들을 더 중요하게 학습하게 되는 것이죠.(사람도 같은 일만 계속하면 문제를 해결한다기보다는 작업량을 줄이는 형태로 꼼수를 쓰죠.)
    학습시에는 loss 가 낮은데, 학습한 적이 없는 검증, 혹은 실제 예측시 loss 가 높아지는 현상이 일어나게 되며, 이를 과적합이라 합니다.
    
- 과적합(Overfitting)
    앞서 XOR 문제를 해결하기 위하여 다층 퍼셉트론을 손수 설계한 적이 있습니다.
    퍼셉트론 하나가 선을 하나 긋는 것이라고 한다면,
    다층 퍼셉트론은 선을 여러개 긋는 것이죠.
    직접 설계한 XOR 문제 해결을 위한 다층 퍼셉트론은,
    선을 딱 두개를 그어서 분류를 했습니다.
    이러면 구조도 단순하고 성능도 좋은 모델이 되겠죠.
    하지만 이번에 한 것과 같이 이미지 분류와 같은 자연적인 데이터를 분류하는 모델은 필연적으로 복잡해질 수 밖에 없습니다.
    선이 여러개 사용된다는 것이죠.
    데이터를 분류 할 때,
    눈에 보이는 덩어리가 두개 확인되면, 그 두 덩어리 사이에 선을 하나 그으면 될 일입니다.
    하지만 딥러닝 학습으로 복잡한 모델을 학습시킬때에는, 학습에 준비된 데이터의 덩어리의 외곽을 아주 촘촘하게 선을 그을 수 있습니다.
    이런식으로 학습된 모델은, 학습시에는 당연히 좋은 결과를 만들어낼 테지만, 실제 사용을 하려고 한다면,
    아주 약간의 노이즈나 오차로인한 틀어짐에도 용납을 하지 않고 다르게 분류를 해낼 것입니다.
    이것이 과적합이죠. (사람으로 치자면 자신이 경험한 특이사항에 대해 편견이 생긴 것을 의미합니다.)
    
- 과적합을 감지하는 방법은,
    학습시 학습용 데이터와 테스트용 데이터를 따로 준비하는 것입니다.
    학습에 사용되는 데이터로 학습을 시킨 후, 테스트용 데이터로 해당 모델을 검증합니다.
    학습을 반복할수록 학습 데이터로 측정한 검증 결과는 개선될 것입니다.
    하지만 학습에 사용되지 않은 테스트용 데이터의 경우는 어느순간 학습을 할수록 검증 결과가 악화될 수 있습니다.
    이렇게 학습시의 검증 결과와 테스트시의 검증 결과가 차이가 나는 부분이 바로 모델이 학습 데이터에 특화되어 과적합되는 순간입니다.

- 과적합을 막는 방법은,
    1. 양질의 학습 데이터를 많이 사용하고, 학습한 데이터에 대한 재활용을 줄일 것
        : 실제 현상을 완벽하게 설명해주는 모든 데이터를 학습에 사용한다고 가정한다면,
        만약 학습 데이터에 굉장히 특화된 모델이 나오더라도 사실상 아무 문제가 없는 모델입니다.
        반면, 실제 현상을 이해하는데 방해가 되는 노이즈가 포함된 학습 데이터를 반복해서 학습하여,
        해당 데이터에 특화된 모델을 만들어낸다면 실제 사용에 방해가 된다는 것을 추론할 수 있습니다.
        
    2. 모델의 복잡도를 줄일 것
        : 앞서 XOR 회로를 다층 퍼셉트론으로 만들 때는 필요 최소한의 복잡도로,
        선형 회귀 모델의 선을 단 두개를 사용한 것과 같은데,
        이처럼 모델의 복잡도가 작다면, 잘못된 방식으로 학습시 효과를 내는 것을 방지할 수 있습니다.
        잘못된 방식을 작성하는데 필요한 자원이 없기에, 그만큼 올바른 학습을 강제할 수 있기 때문이죠.
    
    3. 가중치 규제(Regularization) 을 적용할 것
        : 복잡한 모델은 학습이 진행되며 필요없는 가중치가 존재하기 마련입니다.
        가중치의 값이 100 단위 10000 단위로 늘어날수록 가중치의 수치는 요동치며,
        이러한 가중치의 약간의 변동이 모델의 성능을 크게 낮출 수도 있습니다.
        가중치 규제란, 손실함수에 가중치의 크기를 합하여,
        가중치를 최소화 하는 방향으로 학습이 이루어지도록 유도하는 것으로,
        가중치가 작아지고, 쓸모없는 가중치가 0에 수렴한다면,
        복잡한 모델이라고 하더라도, 잔가지를 제거한 단순한 모델로 다이어트가 되는 효과를 볼 수 있습니다.
    
    4. 드롭아웃을 사용할 것
        : 드롭아웃은, '학습시' 랜덤하게 선택된 가중치의 일부를 사용하지 않는 것을 의미합니다.
        오버피팅이 특정한 패턴과 결합에 지나치게 의존하여 강건함을 잃었다고 하면,
        특정 가중치가 예측에 지나치게 관여한다는 뜻이 됩니다.
        랜덤하게 가중치를 off 시킴으로써 특정 가중치에 의존하지 않도록 하는 효과를 지닙니다.
    
    위와 같습니다.
    각 방법에 대한 적용 방법은 다른 글에 정리하겠습니다.
"""


# [MNIST 분류]
def main():
    # 사용 가능 디바이스
    device = tu.get_gpu_support_device(gpu_support=True)

    # 데이터셋 객체 생성 (ex : tensor([[-10., 100., 82.], ...], device = cpu), tensor([[327.7900], ...], device = cpu))
    # x : (28, 28), y : (1) 형식입니다. y 의 경우는 One Hot Encoding 없이 CrossEntropyLoss 함수에 그대로 넣어줘도 됩니다.
    train_dataset = dsets.MNIST(
        root='../resources/datasets/global/MNIST_data/',
        train=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                lambda x: x.view(-1)  # 이미지를 1차원 벡터로 평탄화
            ]
        ),
        download=True
    )

    validation_dataset = dsets.MNIST(
        root='../resources/datasets/global/MNIST_data/',
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                lambda x: x.view(-1)  # 이미지를 1차원 벡터로 평탄화
            ]
        ),
        download=True
    )

    # 데이터 로더 래핑
    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=10, shuffle=True, drop_last=True)

    # 모델 생성
    model = multi_layer_perceptron_for_mnist.MainModel()

    # 모델 학습
    tu.train_model(
        device=device,
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.SGD(model.parameters(), lr=0.1),
        train_dataloader=train_dataloader,
        num_epochs=15,
        validation_dataloader=validation_dataloader,
        check_point_file_save_directory_path="../_check_point_files/multi_layer_perceptron_for_mnist_nn",
        # check_point_load_file_full_path="../_check_point_files/~/checkpoint(2024_02_29_17_51_09_330).pt",
        log_freq=1
    )

    # 모델 저장
    model_file_save_directory_path = "../_torch_model_files/multi_layer_perceptron_for_mnist_nn"
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


if __name__ == '__main__':
    main()
