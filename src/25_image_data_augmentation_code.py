from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np

"""
[이미지 데이터 증강]
이미지 데이터를 증강시키는 여러 기법을 모아둡니다.

데이터 변형으로 인한 증강 방식을 사용할 때,
변형의 정도가 너무 작다면 동일 데이터를 사용한 것과 같이 오버피팅이 발생하며,
변형의 정도가 너무 크다면 데이터 품질이 저하될 수 있습니다.
"""

img = Image.open("../resources/datasets/25_image_data_augmentation_code/cat.jpg")


# 1.
def transform1(image):
    transform = transforms.Compose(
        [
            # 리사이징
            transforms.Resize(size=(512, 512)),
            # 텐서로 변경
            transforms.ToTensor()
        ]
    )

    # 변환
    transformed_image = transform(image)

    print(transformed_image.shape)

    transformed_image_np = transformed_image.numpy().transpose((1, 2, 0))

    print(transformed_image_np.shape)

    plt.imshow(transformed_image_np)
    plt.show()


# 2.
def transform2(image):
    transform = transforms.Compose(
        [
            # 무작위 회전
            transforms.RandomRotation(degrees=30, expand=False, center=None),
            # 좌우 반전
            transforms.RandomHorizontalFlip(p=0.5),
            # 상하 반전
            transforms.RandomVerticalFlip(p=0.5)
        ]
    )

    # 변환
    transformed_image = transform(image)

    plt.imshow(transformed_image)
    plt.show()


# 3.
def transform3(image):
    transform = transforms.Compose(
        [
            # 무작위 자르기
            transforms.RandomCrop(size=(512, 512)),
            # 패딩 채우기
            transforms.Pad(padding=50, fill=(127, 127, 255), padding_mode="constant")
        ]
    )

    # 변환
    transformed_image = transform(image)

    plt.imshow(transformed_image)
    plt.show()


# 4.
def transform4(image):
    transform = transforms.Compose(
        [
            # 무작위 아핀 변환
            transforms.RandomAffine(
                degrees=15, translate=(0.2, 0.2),
                scale=(0.8, 1.2), shear=15
            )
        ]
    )

    # 변환
    transformed_image = transform(image)

    plt.imshow(transformed_image)
    plt.show()


# 5.
def transform5(image):
    transform = transforms.Compose(
        [
            # 색상 변형
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3,
                saturation=0.3, hue=0.3
            ),
            transforms.ToTensor(),
            # 픽셀 정규화
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.ToPILImage()
        ]
    )

    # 변환
    transformed_image = transform(image)

    plt.imshow(transformed_image)
    plt.show()


# 6.
def transform6(image):
    transform = transforms.Compose(
        [
            # 이미지를 텐서로 변환
            transforms.ToTensor(),
            # 이미지 무작위 구역 지우기(값을 0 으로 채우기)
            # 컷아웃은 동영상에서 폐색 영역(Occlusion) 에 대해 강건하게 해줍니다.
            transforms.RandomErasing(p=1.0, value=0),
            # 이미지 무작위 구역 지우기(값을 랜덤 값으로 채우기)
            # 무작위 지우기는 일부 영역 누락 및 잘렸을 경우에 대해 강건하게 해줍니다.
            transforms.RandomErasing(p=1.0, value='random'),
            # 텐서를 이미지로 변환
            transforms.ToPILImage()
        ]
    )

    # 변환
    transformed_image = transform(image)

    plt.imshow(transformed_image)
    plt.show()


# 7.
class Mixup:
    def __init__(self, target, scale, alpha=0.5, beta=0.5):
        self.target = target
        self.scale = scale
        self.alpha = alpha
        self.beta = beta

    def __call__(self, image):
        image = np.array(image)
        target = self.target.resize(self.scale)
        target = np.array(target)
        mix_image = image * self.alpha + target * self.beta
        return Image.fromarray(mix_image.astype(np.uint8))


def transform7(image):
    transform = transforms.Compose(
        [
            # 이미지 리사이징
            transforms.Resize((512, 512)),
            # 이미지 혼합
            # 다중 레이블 문제에 대해 강건해집니다.
            Mixup(
                target=Image.open("../resources/datasets/25_image_data_augmentation_code/dog.jpg"),
                scale=(512, 512),
                alpha=0.5,
                beta=0.5
            )
        ]
    )

    # 변환
    transformed_image = transform(image)

    plt.imshow(transformed_image)
    plt.show()


transform7(img)
