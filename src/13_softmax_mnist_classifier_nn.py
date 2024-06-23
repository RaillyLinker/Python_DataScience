import torch
from torch import nn
from torch.utils.data import DataLoader
import utils.torch_util as tu
import model_layers.softmax_mnist_classifier.main_model as softmax_mnist_classifier
import os
from torch import optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms

"""
[소프트맥스 회귀 모델]
- 이번 정리글은 소프트맥스 회귀 모델을 사용하여 MNIST 숫자 이미지 분류를 해볼 것입니다.

- One-Hot Encoding 이란?
    원-핫 인코딩은 선택해야 하는 선택지의 개수만큼의 차원을 가지면서,
    각 선택지의 인덱스에 해당하는 원소에는 1, 나머지 원소는 0의 값을 가지도록 하는 표현 방법입니다.

    예를 들어 강아지, 고양이, 냉장고라는 3개의 선택지가 있다고 해보겠습니다.
    원-핫 인코딩을 하기 위해서는 우선 각 선택지에 순차적으로 정수 인덱스를 부여합니다.
    임의로 강아지는 0번 인덱스, 고양이는 1번 인덱스, 냉장고는 2번 인덱스를 부여하였다고 해봅시다.
    이때 각 선택지에 대해서 원-핫 인코딩이 된 벡터는 다음과 같습니다.

    강아지 = [1, 0, 0]
    고양이 = [0, 1, 0]
    냉장고 = [0, 0, 1]

    총 선택지는 3개였으므로 위 벡터들은 전부 3차원의 벡터가 되었습니다.
    그리고 각 선택지의 벡터들을 보면 해당 선택지의 인덱스에만 1의 값을 가지고, 나머지 원소들은 0의 값을 가집니다.
    예를 들어 고양이는 1번 인덱스였으므로 원-핫 인코딩으로 얻은 벡터에서 1번 인덱스만 1의 값을 가지는 것을 볼 수 있습니다.

    이와 같이 원-핫 인코딩으로 표현된 벡터를 원-핫 벡터(one-hot vector)라고 합니다.

    여기서 볼 수 있듯이, 벡터의 각 위치마다 의미를 부여하고, 그에 따른 활성(1) 비활성(0) 으로 표현하는 방식이며,
    이는 다르게 생각하면, 확률로 해석이 가능합니다.

    벡터 안의 모든 값들의 총합을 1이라고 두고,

    [0.8, 0.1, 0.1]

    이렇게 표현한다면, 이는 강아지일 확률이 0.8, 고양이나 냉장고일 확률이 0.1 이라고 해석이 가능한 것으로,
    표현의 명확함과 단순성으로 인하여 상황에 따라 매우 좋은 인코딩 방법이 될 수 있지만,
    의미가 하나 추가될 때마다 컬럼이 하나씩 추가되기에, 데이터 크기가 매우 커질 수 있다는 단점이 있습니다.
    (그냥 커진다기보다는 1 외에는 모두 0으로 채워집니다. 이를 희소하다고 하며, 이런 벡터를 희소 벡터라고 합니다.)

- Softmax Regression 이란?
    이진 분류가 두 개의 답 중 하나를 고르는 문제였다면,
    세 개 이상의 답 중 하나를 고르는 문제를 다중 클래스 분류(Multi-class Classification)라고 합니다.
    독립변수가 주어지면 해당 데이터를 가지고 어떤 클래스에 속하는지 분류하는 것입니다.

    소프트맥스 회귀가 바로 다중 클래스 분류 문제입니다.
    소프트맥스 회귀는 각 클래스. 즉, 각 선택지마다 소수 확률을 할당합니다.
    이때 총 확률의 합은 1이 되어야 합니다. 이렇게 되면 각 선택지가 정답일 확률로 표현됩니다.(앞서 원 핫 벡터에서 설명한 내용입니다.)
    결국 소프트맥스 회귀는 선택지의 개수만큼의 차원을 가지는 벡터를 만들고,
    해당 벡터가 벡터의 모든 원소의 합이 1이 되도록 원소들의 값을 변환시키는 어떤 함수를 지니게 만들어야 합니다.
    이 함수를 소프트맥스(softmax) 함수라고 합니다.

- Softmax 함수
    소프트맥스 함수는 분류해야하는 정답지(클래스)의 총 개수를 k라고 할 때,
    k차원의 벡터를 입력받아 각 클래스에 대한 확률을 추정합니다.
    벡터 내의 k 개의 모든 변수의 값들을 더한 값을 가지고 이를 각 변수마다 나누어주면 되는 것입니다.

- 손실 함수
    소프트맥스 회귀는 이진 분류가 아닌 다중 분류입니다.
    분류의 문제이므로 크로스 엔트로피 함수가 효율적입니다.
    
- MNIST 데이터 
    링크 : http://yann.lecun.com/exdb/mnist
    MNIST는 숫자 0부터 9까지의 이미지로 구성된 손글씨 데이터셋입니다. 
    이 데이터는 과거에 우체국에서 편지의 우편 번호를 인식하기 위해서 만들어진 훈련 데이터입니다. 
    총 60,000개의 훈련 데이터와 레이블, 총 10,000개의 테스트 데이터와 레이블로 구성되어져 있습니다. 
    레이블은 0부터 9까지 총 10개입니다. 
    MNIST 문제는 손글씨로 적힌 숫자 이미지가 들어오면, 그 이미지가 무슨 숫자인지 맞추는 문제입니다. 
    예를 들어 숫자 5의 이미지가 입력으로 들어오면 이게 숫자 5다! 라는 것을 맞춰야 합니다. 
    각각의 이미지는 아래와 같이 28 픽셀 × 28 픽셀의 이미지입니다.
"""

# [Cross Entropy 구현]
# 앞서 이진 분류에서 Binary Cross Entropy 를 구현해봤는데,
# 이번에는 다중 분류에 사용되는 Cross Entropy 를 구현해보겠습니다.

# 3 행 5 열의 랜덤 실수 매트릭스 생성
# ex : tensor([[0.3071, 0.2267, 0.3375, 0.1689, 0.1826],
#         [0.0179, 0.4939, 0.2006, 0.3813, 0.8228],
#         [0.7049, 0.0768, 0.2372, 0.0557, 0.3179]], requires_grad=True)
z = torch.rand(3, 5, requires_grad=True)
# 행마다 Softmax 를 수행 (행 내의 값들을 모두 더하여 1 이 되도록)
# tensor([[0.2124, 0.1960, 0.2190, 0.1850, 0.1876],
#         [0.1336, 0.2151, 0.1604, 0.1922, 0.2988],
#         [0.2975, 0.1587, 0.1864, 0.1554, 0.2020]], grad_fn=<SoftmaxBackward0>)
hypothesis = nn.functional.softmax(z, dim=1)

# 5 열의 벡터 중 정답에 해당하는 인덱스 생성 (3 행이므로 3개 생성)
# ex : tensor([4, 2, 1])
y = torch.randint(5, (3,)).long()

# 모든 원소가 0의 값을 가진 3 × 5 텐서 생성 후 정답 y 의 One Hot Vector 로 만들기
# tensor([[0., 0., 0., 0., 1.],
#         [0., 0., 1., 0., 0.],
#         [0., 1., 0., 0., 0.]])
y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y.unsqueeze(1), 1)

# Cross Entropy 계산
# ex : tensor(1.7815, grad_fn=<MeanBackward0>)
cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost)

# 토치에서 제공하는 손실 함수는 아래와 같이 작성하면 됩니다. (입력값에 주의하세요.)
# ex : tensor(1.7815, grad_fn=<NllLossBackward0>)
cost = nn.functional.cross_entropy(z, y)
print(cost)


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
    model = softmax_mnist_classifier.MainModel()

    # 모델 학습
    tu.train_model(
        device=device,
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.SGD(model.parameters(), lr=0.1),
        train_dataloader=train_dataloader,
        num_epochs=15,
        validation_dataloader=validation_dataloader,
        check_point_file_save_directory_path="../_by_product_files/check_point_files/softmax_mnist_classifier_nn",
        # check_point_load_file_full_path="../_by_product_files/check_point_files/~/checkpoint(2024_02_29_17_51_09_330).pt",
        log_freq=1
    )

    # 모델 저장
    model_file_save_directory_path = "../_by_product_files/torch_model_files/softmax_mnist_classifier_nn"
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
