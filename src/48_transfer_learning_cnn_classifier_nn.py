import torch
from torch import nn
from torch.utils.data import DataLoader
import utils.torch_util as tu
import model_layers.cnn_transfer_classifier.main_model as cnn_transfer_classifier
import os
from torch import optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

"""
[전이학습 (Transfer Learning)]
- 딥러닝에는 학습 데이터가 굉장히 많이 필요합니다.
    성능을 위해 학습해야하는 층이 깊다면 그만큼 파라미터가 많기 때문이죠.
    그렇기에 컴퓨팅 연산 코스트와 시간 역시 많이 들어갑니다.
    이러한 문제점을 해결하기 위하여 이미 잘 학습된 모델의 구조와 파라미터를 가져와 사용하는 방식을 사용할 수 있습니다.
    이를 전이학습이라 부릅니다.

- 아래는 ImageNet 데이터셋으로 학습된 VGGNet 을 가져와 사용하는 예시를 작성할 것입니다.
    ImageNet은 1000 가지 종류로 나뉜 120만 개가 넘는 이미지를 놓고,
    어떤 물체인지를 맞히는 이미지넷 이미지 인식 대회(ILSVRC)에 사용되는 데이터셋입니다.
    전체 크기가 200GB 에 이를 만큼 커다란 데이터이므로 학습이 쉽지는 않지만,
    이 양질의 데이터셋으로 학습한 CNN 모델은 범용적인 이미지 특징을 잘 학습한 모델이라고 생각 할 수 있습니다.

    VGGNet 은 옥스포드 대학의 연구팀 VGG 에 의해 개발된 모델로, 2014년 ILSVRC 에서 2위를 차지한 모델입니다.
    학습 구조에 따라 VGG16, VGG19 등의 이름이 지어졌는데, 여기서는 VGG16을 사용할 것입니다.
"""


def main():
    # 사용 가능 디바이스
    device = tu.get_gpu_support_device(gpu_support=True)

    # 데이터셋 객체 생성 (ex : tensor([[-10., 100., 82.], ...], device = cpu), tensor([[327.7900], ...], device = cpu))
    # x : (28, 28), y : (1) 형식입니다. y 의 경우는 One Hot Encoding 없이 CrossEntropyLoss 함수에 그대로 넣어줘도 됩니다.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_dataset = datasets.CIFAR10(root='../resources/datasets/global', train=True, transform=transform,
                                     download=True)

    validation_dataset = datasets.CIFAR10(root='../resources/datasets/global', train=False, transform=transform,
                                          download=True)

    # 데이터 로더 래핑
    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=10, shuffle=True, drop_last=True)

    # 모델 생성
    model = cnn_transfer_classifier.MainModel()

    # 모델 학습 (== VGG 모델에 대한 전이학습)
    tu.train_model(
        device=device,
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.SGD(model.parameters(), lr=0.001, momentum=0.9),
        train_dataloader=train_dataloader,
        num_epochs=2,
        validation_dataloader=validation_dataloader,
        check_point_file_save_directory_path="../_check_point_files/cnn_transfer_classifier",
        # check_point_load_file_full_path="../_check_point_files/~/checkpoint(2024_02_29_17_51_09_330).pt",
        log_freq=1
    )

    # 모델 저장
    model_file_save_directory_path = "../_torch_model_files/cnn_transfer_classifier"
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
- 위 코드로, 미리 학습된 VGG16 을 가져와서 사용하는 예시를 알아보았습니다.
    모델 구조는, VGG 모델에서, CNN 레이어들을 거쳐서 이미지의 특징을 출력하는 역할을 하는 features 레이어에 이미지가 들어가고,
    그 결과를 pooling 한 후,
    그렇게 얻어낸 특징맵을 Flatten 하여, 은닉층으로 10개 클래스로 분류하도록 하였습니다.
    (특징 추출 -> Pooling -> Flatten -> 은닉층)
    
    CNN 을 사용하는 모델을 설계할 때에는 대부분 이 구조 그대로 적용 하면 됩니다.
    
- 위 학습에서는 CIFAR-10 데이터셋을 사용하였는데, 각 클래스는 다양한 사물이나 객체를 대표합니다. 
    CIFAR-10 데이터셋의 클래스는 다음과 같습니다.
    
    1. 비행기 (airplane)
    2. 자동차 (automobile)
    3. 새 (bird)
    4. 고양이 (cat)
    5. 사슴 (deer)
    6. 개 (dog)
    7. 개구리 (frog)
    8. 말 (horse)
    9. 배 (ship)
    10. 트럭 (truck)

- 위 분류기 모델은, 
    모델 클래스 내에서 param.requires_grad = False 라는 코드로 VGG 관련 파라미터를 동결시키고,
    뒤에 연결된 분류용 레이어만 학습을 시켰습니다.
    
"""
