"""
[설명 가능한 AI (Explainable AI :XAI)]
- 딥러닝은 매우 효과적으로 답을 내어놓는 AI 모델인데, 내부 은닉층의 존재와 스스로 학습하는 특성으로 인해 그 결과를 설명하지 못하는 단점이 있습니다.
    자연어 처리가 발달하면 모델이 자체적으로 근거를 자연어로 생성하기에 이 부분은 많이 해소될 테지만,
    경량화를 위하여 결과만을 출력하는 분석 모델의 경우는 왜 모델이 그러한 결과를 냈는지 알기 어려우며,
    잘못된 결과에 대한 리스크와, 그에따른 설득력의 저하와 같은 문제가 있습니다.
    XAI 는 단순히 답을 내는 것 뿐 아니라, 그 근거를 제시 하도록 하는 방식을 의미 합니다.

- 컴퓨터 비전 분야에서 사용되는 CNN 모델에서 대표적으로 사용되는 XAI 는 CAM(Class Activation Map)이라는 것이 있습니다.
    이미지 분류 모델이 해당 입력 이미지가 '강아지'라고 판단하였다면,
    이미지의 어느 부위가 '강아지'라는 결정을 내는 데에 큰 영향을 준 것인지에 대해 출력해주는 방식으로,
    이미지와 동일한 크기의 맵을 만들어서, 최종 결정에 영향을 많이 끼치는 부분에 큰 점수를 주고, 아니면 낮은 점수를 주는 방식으로 결과에 대한 근거를 제시합니다.

- CAM 의 원리를 알아보겠습니다.
    CNN 분류 모델의 경우, 기본 구조는,
    CNN 특징 추출 레이어로 특징맵 추출 -> 특징맵을 Flatten 하여 벡터화 -> 특징 벡터를 Classification 하는 은닉층으로 결과 출력
    위와 같습니다.
    이때 우리가 알고 싶은 것은, 이미지의 어느 부분이 최종 분류 은닉층에 영향을 끼치는지에 대한 히트맵을 얻는 것입니다.
    CAM 을 적용하기 위하여 모델의 구조를 조금 바꾸겠습니다.
    CNN 특징 추출 레이어로 특징맵 추출 -> 출력된 특징맵 N 개의 개별 맵들마다 평균값을 가져오기 -> 특징 벡터를 Classification 하는 은닉층으로 결과 출력
    위와 같습니다.
    바뀐 부분이란, Flatten 레이어를 적용하는 부분을, "출력된 특징맵 N 개의 개별 맵들마다 평균값을 가져오기" 이것으로 바꾼 것 뿐인데,
    이는 GAP(Global Average Pooling) 이라고 부릅니다.
    특징맵이란, 이미지의 특성이죠?
    CNN 레이어 끝에서 반환되는 특징맵들은, 이미지를 분류하기 위해 필요한 특징들에 대한 평가값을 행렬로 표현한 것입니다.
    예를들어 특징맵 A 가 이미지의 거친 질감을 의미하고, 특징맵 B 가 이미지의 투명함을 의미할 때,
    특징맵 A 의 각 값들은, 이미지의 구역별 거친 질감의 정도에 대한 값을 표현해주고,
    특징맵 B 의 각 값들은, 이미지의 구역별 투명한 정도에 대한 값을 표현해주는 것입니다.
    GAP 를 적용한다는 것은, 이미지의 전체적인 거친 질감과, 전체적인 투명한 정도를 평균내어 구한다는 뜻입니다.
    이 결과를 종합하여 분류 결과를 구한다면,
    각 특징맵들마다 분류 결과에 영향을 끼치는 정도를 알 수 있겠네요. (해당 위치에 연결된 파라미터)
    영향력을 알아내는 법을 알게되었으므로, 실제 픽셀에 이를 투사해야합니다.
    이때는 GAP 를 하기 전의 특징맵을 사용합니다.
    특징맵은 축소되기는 했지만, 이미지의 위치적 특성이 보존되어 있습니다.
    모든 특징맵의 각 값마다, 해당하는 가중치를 모두 곱해주고, 특징맵을 모두 더해주면 됩니다.
    이렇게 하면, 높은 가중치가 나오는 특징맵의 값들이 부각되고, 낮은 가중치의 특징맵의 값은 낮아지며,
    특징맵 내에서 높은 값이 나오는 픽셀 구역이 높은 값을 나타내게 되므로,
    이것이 바로 해당 모델이 왜 이러한 분류 결과를 내었는지에 대한 근거가 되는 것입니다.
"""

import os
from torch import optim
from torch.utils.data import DataLoader
import utils.torch_util as tu
import model_layers.cnn_classifier_cam.main_model as cnn_classifier_cam
import torchvision.datasets as datasets
import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# Class Activation Mapping 생성
def generate_cam(
        # CNN 분류 모델
        image_classifier_cnn_model,
        # CNN 분류 모델 입력 이미지
        input_image,
        # 분류 클래스 인덱스 (None 이라면 분류 결과가 가장 유력한 클래스)
        target_class_idx=None
):
    # 모델을 검증 모델로 변경
    image_classifier_cnn_model.eval()

    # 마지막 컨볼루션 레이어와 FC 레이어의 가중치 찾기
    last_conv_layer = None
    fc_layer_weights = None

    # 모델의 레이어를 역순으로 탐색하여 마지막 컨볼루션 레이어와 FC 레이어의 가중치를 찾습니다.
    for layer in image_classifier_cnn_model.modules():
        if isinstance(layer, nn.Conv2d):
            last_conv_layer = layer
        elif isinstance(layer, nn.Linear):
            fc_layer_weights = layer.weight
            break

    # 피쳐 맵을 저장할 변수
    # feature_map 의 차원은 (배치 크기, 채널 수, 높이, 너비)
    feature_map = None

    # 후크를 이용하여 마지막 CNN 레이어에서 나오는 피쳐 맵을 저장
    def forward_hook(module, input, output):
        nonlocal feature_map
        feature_map = output

    # CNN 마지막 레이어에 후크 등록
    hook_handle = last_conv_layer.register_forward_hook(forward_hook)

    # 이미지 추론
    output = image_classifier_cnn_model(input_image)

    # 후크 제거
    hook_handle.remove()

    # 예측 결과 텐서에서 가장 높은 값을 가진 요소와 해당 요소의 인덱스 반환
    max_value, max_index = torch.max(output, dim=0)

    if target_class_idx is None:
        target_class_idx = max_index

    print(f"가장 높은 값을 가진 클래스 인덱스: {max_index.item()}")
    print(f"가장 높은 값: {max_value.item()}")

    print(f"선택한 클래스 인덱스: {target_class_idx}")
    print(f"선택한 클래스 값: {output[target_class_idx]}")

    # feature_map 의 높이와 너비 만한 빈 이미지 결과맵 생성
    result_cam = torch.zeros(feature_map.shape[2], feature_map.shape[3])

    # 추출된 feature_map 들을 순회
    for i in range(feature_map.shape[1]):
        result_cam += fc_layer_weights[target_class_idx, i] * feature_map[0, i, :, :]

    # CAM을 원래 이미지의 크기로 변환
    result_cam = f.interpolate(result_cam.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear',
                               align_corners=False)
    result_cam = result_cam.squeeze().detach().numpy()

    # Normalize CAM
    result_cam = (result_cam - result_cam.min()) / (result_cam.max() - result_cam.min())

    return result_cam


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
    model = cnn_classifier_cam.MainModel()

    # !!!학습이 필요하면 주석을 푸세요.!!!
    # # 모델 학습 (== VGG 모델에 대한 전이학습)
    # tu.train_model(
    #     device=device,
    #     model=model,
    #     criterion=nn.CrossEntropyLoss(),
    #     optimizer=optim.SGD(model.parameters(), lr=0.001, momentum=0.9),
    #     train_dataloader=train_dataloader,
    #     num_epochs=1,
    #     validation_dataloader=validation_dataloader,
    #     check_point_file_save_directory_path="../_check_point_files/cnn_classifier_cam",
    #     # check_point_load_file_full_path="../_check_point_files/~/checkpoint(2024_02_29_17_51_09_330).pt",
    #     log_freq=1
    # )
    #
    # # 모델 저장
    # model_file_save_directory_path = "../_torch_model_files/cnn_classifier_cam"
    # if not os.path.exists(model_file_save_directory_path):
    #     os.makedirs(model_file_save_directory_path)
    # save_file_full_path = tu.save_model_file(
    #     model=model,
    #     model_file_save_directory_path=model_file_save_directory_path
    # )

    # # 저장된 모델 불러오기
    model_path = "../resources/datasets/49_xai_cam/cnn_classifier_cam_model/model(2024_04_15_22_50_06_450).pt"
    model = torch.load(model_path, map_location="cpu")

    # 이미지 경로
    image_path = '../resources/datasets/49_xai_cam/cat.jpg'
    # 이미지 불러오기
    original_image = Image.open(image_path)

    # 이미지 전처리
    input_tensor = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet 모델은 224x224 크기의 이미지를 필요로 합니다.
        transforms.ToTensor(),  # 이미지를 텐서로 변환
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
    ])(original_image)

    # 배치 차원을 추가하여 모델에 입력할 수 있는 형태로 변경
    input_tensor = input_tensor.unsqueeze(0)  # (1, 3, 224, 224) 형태로 변경

    # CAM 생성
    cam = generate_cam(model, input_tensor, target_class_idx=None)

    # CAM 을 0~255 사이의 값으로 조정
    cam_heatmap = np.uint8(255 * cam)

    # CAM 으로 히트맵 생성
    cam_heatmap = plt.cm.jet(cam_heatmap)[:, :, :3]

    # 원본 이미지를 CAM 히트맵 크기로 리사이즈
    resized_image = original_image.resize((224, 224))

    # 원본 이미지를 numpy 배열로 변환
    resized_image_array = np.array(resized_image) / 255

    # CAM 히트맵과 원본 이미지의 합성
    combined_image = 0.5 * resized_image_array + 0.5 * cam_heatmap

    # 결과를 시각화
    plt.imshow(combined_image)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
