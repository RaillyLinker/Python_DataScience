import torch
from torch import nn
from torch.utils.data import DataLoader
import utils.torch_util as tu
import model_layers.cnn_mnist_classifier.main_model as cnn_mnist_classifier
import os
from torch import optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms

"""
[합성곱 신경망 모델]
- 앞서 MNIST 손글씨 이미지 분류기를 만들어본 적이 있습니다.
    CNN 은 바로 그러한 이미지 분석에 효율적인 신경망 모델입니다.
    
- 먼저 이미지라는 것에 대해 알아보겠습니다.
    시각 정보에서 형태라는 것은 2차원 행렬으로 나타냅니다.
    x축과 y 축으로 이루어진 행렬맵 위에, 이미지 정보가 표현됨으로써 사람이 보는 시각 정보의 형태가 이루어집니다.
    가장 기본적인 시각 정보로 흑백 정보를 표현한다고 하면, 
    이 2차원 행렬맵 위에, 검은색을 뜻하는 0에서부터 하얀색을 뜻하는 255 사이의 값을 표현하는 방식으로 표현이 가능합니다.
    
    0 0 0 0 0 0 0
    0 0 7 7 7 0 0
    0 0 7 0 0 0 0
    0 0 7 7 0 0 0
    0 0 0 0 7 0 0
    0 0 0 0 7 0 0
    0 0 7 7 0 0 0
    
    예를들어 위의 데이터는 숫자 5 를 표현한 것입니다.
    검은 바탕에, 7 이라는 데이터가 5 라는 형태로 표현된 것입니다.
    이외에, 흑백뿐 아니라, 정말 사람이 보는 컬러 데이터를 표시하려면, RGB, 혹은 YUV 와 같은 데이터 표시 방법이 있는데,
    실상은 형태를 나타내는 위와 같은 행렬맵 위에, 각 위치의 점이 어떤 색을 띄는지를 (Red : 255, Green : 64, Blue : 15) 이런식으로 표현하는 것입니다.
    
- 컨볼루션 신경망(Convolution Neural Network : CNN) 이란,
    생명체의 시각 피질의 구조를 모방한 신경망 모델입니다.
    
    1959 년, David H. Hubel 과 Torsten Wiesel 이 뇌의 시각 피질이 어떻게 동작하는지를 확인하기 위하여,
        마취된 고양이의 일차 시각 피질에 미세 전극을 넣었습니다.
        이를 통하여 시각 피질이 여러 층으로 구성되어 있고, 첫번째 층은 주로 모서리와 직선 감지, 
        뒤쪽 층은 복잡한 모양과 패턴을 추출하는데 사용 된다는 것이 발견되었습니다. 
        (다층으로, 앞의 층일수록 단순한 형태에 반응하고, 뒤로 갈수록 앞의 단순한 형태를 조합한 복잡한 형태에 반응)
        
    1990 년, Yann LeCun 이 손글씨 숫자 분류를 위하여 시각 피질의 구조를 모방한 새로운 신경망 구조로 발표 하였습니다.
    
    2019년, Yann Lecun 은 CNN 의 발명으로 인하여 인공지능 분야에 대한 기여를 인정받아, 컴퓨터 과학 분야에서 가장 영애로운 상인 Turing Award 를 수상하였습니다.
    
- Convolutional Kernel 과 시각 분석 원리
    점이 모여 여러 형태의 선이 되고, 선이 모여 여러 형태의 면이 되고, 면 내에서 자잘한 점이나 선들이 결합하여 여러 질감이 표현되고,
    질감이나 선, 점 등의 종합적인 데이터가 모여서, 비로소 인간이 실제로 눈으로 보는 정보들을 표현할 수 있을 것입니다.
    (멀티모달, 공감각, 3차원 분석, 시각 데이터의 분류에 기억에 저장된 데이터, 언어로 학습된 데이터가 영향을 끼치는 등의 더 깊은 고찰도 필요합니다.)
    
    먼저, 앞서 예시를 들었듯, 기본적인 선을 감지하는 알고리즘은 어떻게 구현하면 좋을까요?
    
    이미지 내에서 어떤 선이 어디에 배치되었는지를 확인한다고 합시다.
    순차적으로 이미지 내의 모든 픽셀을 순회하더라도 무엇이 직선인지 뭔지 모를 것입니다.
    점을 감지하는 그 시점에는 그것은 점일 뿐이지 형태를 알 수 없으니까요.
    
    선인지 뭔지를 감지하려면, 점'들'의 형태를 파악해야합니다.
    즉, 하나의 점이 선에 포함되는지 면에 포함되는지를 알기 위해서는, 그 점뿐만이 아니라 그 점의 인접한 위치의 다른 점들까지 확인해야만 하는 것입니다.
    
    점의 주변으로 다른 점들을 확인하여 해당 점이 어디에 속하는지 알기 위해서는 구역을 떼어내야 합니다.
    예를들어 점을 중심으로 3x3 구역을 확인하여 해당 점이 선인지 아닌지를 판별하는 것입니다.
    
    그리고 동일한 방식으로 다음 픽셀, 다음 픽셀을 판별해나가면,
    
    이미지 데이터는 색 데이터의 모음이 아니라, '선', '선이 아님' 이라는 분류 결과의 데이터 행렬이 될 것입니다.
    이렇게 주변 구역을 포함한 계산을 하기 위하여 준비된 n x m 크기의 필터를 컨볼루션 필터, 혹은 커널이라고 부릅니다.
    
    커널의 계산법은 행렬곱입니다.
    예를들어보겠습니다.
    
    1 0 0 0
    0 1 0 0
    0 0 1 0
    
    위와 같은 이미지가 있다고 할 때,
    여기서 대각선을 추출해본다고 하면,
    
    1 0 0
    0 1 0
    0 0 1
    
    이라는 필터를 생각해보면 됩니다.
    좌측 상단의 픽셀부터 적용하면,
    필터가 닿지 않는 공간의 데이터를 0이라고 두고,
    
    (1 * 0) + (0 * 0) + (0 * 0) + 
    (0 * 0) + (1 * 1) + (0 * 0) + 
    (0 * 0) + (0 * 0) + (1 * 1) = 2
    
    (1 * 0) + (0 * 1) + (0 * 0) + 
    (0 * 0) + (1 * 0) + (0 * 1) + 
    (0 * 0) + (0 * 0) + (1 * 0) = 0
    
    ...
    
    위와 같이 픽셀을 하나씩 이동하며 커널로 행렬곱을 하여 결과를 알아내면,
    
    2 0 0 0
    0 3 0 0
    0 0 2 0
    
    위와 같이 결과가 나옵니다.
    커널의 연산 결과 나온 이 결과 행렬을 컨볼루션(합성곱) 층이라고 부릅니다.
    위 컨볼루션 층은 대각선의 형태를 띈 픽셀 부분의 경우 큰 수가 나오고, 아니라면 작은 수가 나오게 표현된 것이죠.
    
    이외에도 다양한 각도의 직선에 대한 커널을 만들 수 있습니다.
    물론, 컨볼루션 층을 쌓아서 선보다 더 복잡한 형태를 추출해내는 커널을 만들 수도 있고요.
    
    CNN 의 커널의 원리는 위와 같습니다.
    
    CNN 은 딥러닝 모델답게, 오차역전파를 이용하여 커널까지 자동으로 학습을 시킬 수 있습니다.
    즉, 수동으로 커널을 만들어낼 필요가 없으며, 학습시 사용한 정답에 어울리는 은닉된 커널을 스스로 만들어낼 수 있으며,
    데이터만 충분하다면 판별하지 못하는 이미지가 없다는 뜻이 됩니다.
    
- CNN 레이어의 주요한 설정 변수는,
    1. kernel_size : 앞서 설명한 커널의 사이즈입니다.
        커널 크기가 크다면 하나의 판단을 위해 주변 픽셀을 많이 확인한다는 것입니다.
    2. stride : 커널 계산시 모든 픽셀을 순회하려면 이것이 1 이 되고, 3 픽셀마다 한번씩 계산하려면 이것이 3 이 됩니다.
        굳이 모든 픽셀을 순회하기보다는 띄엄띄엄 이미지 내의 특성을 파악하여, 대충 이미지가 어떤 클래스에 속하는지을 알아보는 것 뿐이라면,
        stride 를 적절히 크게 주어서 연산량을 줄이는 것이 좋습니다.
    3. padding & padding_mode : 이미지 외곽에 여분을 주는 것입니다.
        앞선 예시에서 커널이 이미지 외곽 부분을 확인할 때, 이미지가 없을 경우 0 으로 채워넣었었죠.
        바로 이러한 용도의 설정입니다.
        padding_mode 는 패딩을 한 이미지 수치를 0으로 줄지 뭘로 줄지를 결정하는 것입니다.

- Pooling
    이미지는 희소한 데이터가 많습니다.
    무슨 의미냐면, 의미없는 데이터량이 많다는 것입니다.
    예를들어 어떤 이미지가 고양이인지 아닌지를 확인할 때, 이미지 내의 모든 픽셀이 전부 필요하지는 않습니다.
    이미지의 상당한 부분을 가리더라도 고양이인지 아닌지에 대한 판단에는 별 영향이 없을 수도 있죠.
    풀링은 연산량을 줄이기 위하여, 컨볼루션 층을 몇가지 구역으로 나누어서 각 구역별 대표적인 값을 추출함으로써 연산량을 줄이는 방법입니다.
    커널 계산에서 Stride 를 두는 것과 비슷한 효과입니다.
    Sub Sampling 이라고도 불리죠.
    
    구역 내에서 대표값을 최대값으로 두는 것을 Max Pooling 이라고 합니다.
    이 경우는 구역 내에서 가장 특징적인 값을 사용하는 것과 같은 의미이므로 보통 이것을 많이 사용합니다.
    
    이외에는 평균값을 사용하는 Average Pooling 가 존재합니다.
    
- 아래는 합성곱 신경망 CNN 레이어를 사용하여 MNIST 손글씨 이미지 분류기를 만드는 예시입니다.
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
        transform=transforms.ToTensor(),
        download=True
    )

    validation_dataset = dsets.MNIST(
        root='../resources/datasets/global/MNIST_data/',
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )

    # 데이터 로더 래핑
    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=10, shuffle=True, drop_last=True)

    # 모델 생성
    model = cnn_mnist_classifier.MainModel()

    # 모델 학습
    tu.train_model(
        device=device,
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.SGD(model.parameters(), lr=0.1),
        train_dataloader=train_dataloader,
        num_epochs=15,
        validation_dataloader=validation_dataloader,
        check_point_file_save_directory_path="../_check_point_files/cnn_mnist_classifier",
        # check_point_load_file_full_path="../_check_point_files/~/checkpoint(2024_02_29_17_51_09_330).pt",
        log_freq=1
    )

    # 모델 저장
    model_file_save_directory_path = "../_torch_model_files/cnn_mnist_classifier"
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

"""
- 여기까지, 총 3 번을 MNIST 문제 해결 모델을 만들어 봤습니다.
    처음은, 기본적인 자동 학습 기능을 가진 단층 퍼셉트론으로 하여 0.273 정도의 loss 를 가지는 모델을,
    다음은 xor 문제에 대응 가능한 다층 인공 신경망 모델을 사용하여 0.076 정도의 loss 를 가지는 모델로 개선하였으며,
    이번에는 데이터의 위치 정보까지 활용하는 CNN 을 사용한 합성곱 신경망을 사용하여 0.021 정도의 loss 를 가지는 모델까지 발전시킬 수 있었습니다.
    비교적 단순하고 크기도 작은 MNIST 이미지가 아니라, 보다 크고 복잡한 이미지를 사용할 시에는 격차는 더욱 벌어질 것입니다.
    이처럼 결과를 보다시피 CNN 모델은 이미지 정보 분석에 매우 탁월한 성능을 보입니다.
    CNN 이 처음 주목을 받은 이후, 기존의 이미지 분석 모델들의 성능을 능가하기 시작하며,
    이후 계속해서 발전하여, 현재는 이미지 분류쪽으로는 기존 분석 모델은 물론, 사람의 인지 능력을 능가한다고 합니다.
    여러 개선 방식이 존재하겠지만, 결론적으로 CNN 레이어의 깊이가 깊어질수록 성능은 더더욱 나아집니다.
    (물론 파라미터가 많아지고 층이 깊어질수록 양질의 데이터가 필요하며, 학습시 확률적 요소가 커집니다.)
    지금은 이것을 사용하여 이미지 분류 문제에 사용하였지만, CNN 의 가장 큰 의의는 이미지 정보의 '압축'이라고 저는 생각합니다.
    특정 사진을 특정 객체라고 분류가 가능할 정도로 압축된 데이터와, 그 압축 모델은, 그 자체로 쓰이는 것 외에도,
    다른 모델에 특정 이미지의 정보를 입력하는 데이터 전처리기로서 사용하기 좋습니다.
    이렇게 잘 학습된 기존 CNN 분류 모델에서 인코딩 부분만을 떼어내어 다른 모델에 접합 시켜 사용하는 것을 전이학습이라 부르며,
    이 CNN 부분을 백본(BackBone) 모델이라고도 부릅니다.
"""
