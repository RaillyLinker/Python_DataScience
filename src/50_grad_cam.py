import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

"""
[Grad-CAM (Gradient-weighted Class Activation Map)]
- 앞서 정리한 CAM 과 동일한 기술이라고 보아도 좋습니다.
    다른점을 알아보자면,
    CAM 의 경우는 모델의 구조가 정형화 되어있습니다.
    GAP 으로, 각 특징맵별 분류 결과에 끼치는 영향력을 단순화해 놓은 CAM 전용 네트워크에서만 적용이 되며, 
    이 경우 주 기능이 되는 분류 기능에 악영향을 끼칠 수 있습니다.
    반면, Grad-CAM은 일반적인 CNN 구조에 적용할 수 있으며, 
    원래 모델의 성능에 영향을 주지 않는다는 장점이 있습니다.
    
- Grad-CAM 의 원리를 정리하겠습니다.
    일반적으로 CAM 과 기본 원리가 같습니다.
    특징에 대한 중요도를 특징맵의 값에 곱하고, 맵의 동일 위치의 결과값들을 더하는 작업입니다.
    이때, 특징에 대한 중요도를 구할 때, CAM 은 GAP 을 사용한 구조로 바로 뒤의 Grad 를 사용하면 되었는데,
    Grad-CAM 에서는 뒤에 어떠한 구조가 또 올지 알 수 없으므로,
    최종 결과에 끼치는 영향력을 '오차 역전파'를 사용하여 구해야 합니다.
    오차역전파법으로 각 특징맵별 영향력에 대한 기울기값을 구할수 있으므로,
    이 값을 사용하면 CAM 과 원리가 동일하다고 볼 수 있습니다.
    
- CAM 과 비슷한 이미지 분류 모델 해석 기법으로는,
    폐쇄성 민감도(Occlusion Sensitivity) 방식이 존재합니다.
    이미 잘 학습된 모델에서, 이미지를 입력하여 포워딩합시다.
    그러면 분류 결과값이 나올 것입니다.
    다시 동일 모델에 동일 이미지를 다시 입력하면 이전과 동일한 결과가 나오겠죠?
    그런데, 동일 모델에, 동일 이미지를 넣어줄 때, 해당 이미지를 N 분할 하여,
    각 구역을 검은색으로 가려줍니다.
    그러면 출력값에 변화가 있을 것입니다.
    모든 구역을 한번씩 가려보고, 출력값의 변화를 확인하면,
    어떤 구역을 가렸을 때, 결과값이 10 낮아진다면, 해당 부분이 결과에 끼치는 긍정적 영향이 10 이라 할 수 있습니다.
    이런 방식으로 구역 필터링과 그 결과의 변화를 가지고 이미지 분류 모델을 해석하는 방식이 폐쇄성 민감도입니다.
"""


# Grad-CAM 생성
def generate_grad_cam(
        # CNN 분류 모델
        image_classifier_cnn_model,
        # CNN 분류 모델 입력 이미지
        input_image,
        # 분류 클래스 인덱스 (None 이라면 분류 결과가 가장 유력한 클래스)
        target_class_idx=None
):
    # 모델을 검증 모델로 변경
    image_classifier_cnn_model.eval()

    # 마지막 컨볼루션 레이어와 피처 맵 저장할 변수
    last_conv_layer = None
    feature_map = None
    gradients = None

    # 후크를 이용하여 마지막 컨볼루션 레이어에서 나오는 피쳐 맵을 저장하고, 기울기를 계산
    def forward_hook(module, input, output):
        nonlocal feature_map
        feature_map = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    # 마지막 컨볼루션 레이어 찾기
    for layer in image_classifier_cnn_model.modules():
        if isinstance(layer, nn.Conv2d):
            last_conv_layer = layer

    # 마지막 컨볼루션 레이어에 후크 등록
    forward_hook_handle = last_conv_layer.register_forward_hook(forward_hook)

    # backward hook 등록
    backward_hook_handle = last_conv_layer.register_full_backward_hook(backward_hook)

    # 모델 추론
    output = image_classifier_cnn_model(input_image)

    # 예측 결과 텐서에서 가장 높은 값을 가진 요소와 해당 요소의 인덱스 반환
    max_value, max_index = torch.max(output, dim=1)

    if target_class_idx is None:
        target_class_idx = max_index

    print(f"가장 높은 값을 가진 클래스 인덱스: {max_index.item()}")
    print(f"가장 높은 값: {max_value.item()}")

    print(f"선택한 클래스 인덱스: {target_class_idx}")
    print(f"선택한 클래스 값: {output[0][target_class_idx]}")

    # 선택한 클래스에 대한 손실 계산
    loss = output[:, target_class_idx]

    # 역전파 수행
    loss.backward(retain_graph=True)

    # 후크 제거
    forward_hook_handle.remove()
    backward_hook_handle.remove()

    # 기울기를 기반으로 각 채널의 중요도 계산
    weights = torch.mean(gradients, dim=(2, 3))

    # Grad-CAM 계산
    grad_cam = torch.zeros_like(feature_map[0, 0, :, :])
    for i in range(feature_map.shape[1]):
        grad_cam += weights[0, i] * feature_map[0, i, :, :]

    # ReLU를 사용하여 음수 제거
    grad_cam = torch.relu(grad_cam)

    # Grad-CAM을 원래 이미지의 크기로 변환
    grad_cam = f.interpolate(grad_cam.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    grad_cam = grad_cam.squeeze().detach().numpy()

    # Normalize Grad-CAM
    grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())

    return grad_cam


# CNN 모델 로드 (GAP 을 적용하지 않은 원본 모델입니다.)
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# 이미지 경로
image_path = '../resources/datasets/50_grad_cam/cat_and_dog.jpg'
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

# Grad-CAM 생성
# Dog 클래스 인덱스인 204 를 선택하면 강아지 부분이 밝아지고,
# Cat 클래스 인덱스인 281 를 선택하면 고양이 부분이 밝아집니다.
# 그리고 None 을 선택하면 모델이 생각하는 가장 유력한 클래스를 선택합니다.
grad_cam = generate_grad_cam(model, input_tensor, target_class_idx=281)

# Grad-CAM을 0~255 사이의 값으로 조정
grad_cam_heatmap = np.uint8(255 * grad_cam)

# Grad-CAM으로 히트맵 생성
grad_cam_heatmap = plt.cm.jet(grad_cam_heatmap)[:, :, :3]

# 원본 이미지를 Grad-CAM 히트맵 크기로 리사이즈
resized_image = original_image.resize((224, 224))

# 원본 이미지를 numpy 배열로 변환
resized_image_array = np.array(resized_image) / 255

# Grad-CAM 히트맵과 원본 이미지의 합성
combined_image = 0.5 * resized_image_array + 0.5 * grad_cam_heatmap

# 결과를 시각화
plt.imshow(combined_image)
plt.axis('off')
plt.show()
