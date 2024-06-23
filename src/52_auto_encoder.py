"""
[오토 인코더 (Auto Encoder : AE)]
- 인코딩이란, 어떠한 데이터를 특정한 법칙에 따라 가공하는 작업을 말합니다.
    보통 데이터 압축에 사용되며,
    대표적으로, 영상 데이터의 저장 용량을 줄이고, 효율적으로 재생할 수 있도록 하는 비디오 인코딩 기술이 있습니다.
    그렇다고 하여 무조건 인코딩은 압축 기술이라고 단순하게 생각할 수는 없습니다.
    예를들어, 자연어 처리에서 단어 인코딩이란,
    해당 단어를 나타내는 문장의 집합이 아니라, 그 의미를 하나의 벡터로 나타내고자 하는 목적이므로,
    단어를 입력하면 오히려 데이터의 양이 늘어납니다.
    (사과 라는 단어는 단 두글자로 표현이 가능하지만, 빨갛다, 세콤하다, 달다, 맛있다, 과일이다, 단단하다...
    이런식으로 사과라는 단어가 지닌 의미를 모두 품은 벡터값으로 인코딩 하는 것입니다.)
    인코딩과 반대의 의미로는 디코딩이 있으며, 이는 인코딩 된 데이터를 원래대로 복원하는 것을 의미합니다.

- 오토 인코더란,
    인코딩을 자동으로 해주는 딥러닝의 알고리즘입니다.

    입력층으로 값을 넣어줬을 때, 은닉층으로 모종의 연산을 수행한 이후,
    출력층으로는 입력받은 데이터를 그대로 출력하는 것을 목적으로 학습을 시킨 모델이 바로 오토 인코더인데,

    입력층 -> 은닉층

    의 부분이 인코더이고,

    은닉층 -> 출력층

    의 부분이 디코더라고 할 수 있습니다.

    즉, 은닉층이 만들에낸 벡터는 출력층을 거쳐서 다시 원본으로 복구가 가능한 데이터이므로, 인코딩한 결과물이 되는 것입니다.

    딥러닝 알고리즘의 특성상, 은닉층의 결과물은 개발자가 직접적으로 학습시킬 수 없고, 시킬 이유도 없습니다.
    그저 입력값과 정답값만 준비를 하면 위와 같은 구조에서 인코더와 디코더를 자동으로 만들어주는 것이죠.
    이때, 입력값은 비교적 손쉽게 구할 수 있고, 정답값의 경우는 입력값이 주어진 시점에 손에 들어온 것이기에 더욱 편리하며,

    아마 이렇게 편리하고 자동적으로 인코더를 만들어주기에 Auto 인코더라는 이름이 붙은 것이라고 생각합니다.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms

# 데이터셋과 데이터 로더 설정
train_dataset = datasets.MNIST(
    root='../resources/datasets/global/MNIST_data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.RandomRotation(10),  # 데이터 증강: 랜덤 회전 적용
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 데이터 증강을 적용한 MNIST 테스트 데이터셋 로드
test_dataset = datasets.MNIST(
    root='../resources/datasets/global/MNIST_data',
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.RandomRotation(10),  # 데이터 증강: 랜덤 회전 적용
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# AutoEncoder 클래스 정의
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # 인코더 정의
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5)  # 드롭아웃 추가
        )

        # 디코더 정의
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# 모델을 인스턴스화합니다.
model = AutoEncoder()

# 손실 함수와 옵티마이저 설정
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, _ in train_loader:
        # 입력 데이터를 28x28 크기에서 784 크기로 펼칩니다.
        inputs = inputs.view(inputs.size(0), -1)

        # 옵티마이저의 그라디언트를 초기화합니다.
        optimizer.zero_grad()

        # 모델의 순전파
outputs = model(inputs)

# 손실 계산
loss = criterion(outputs, inputs)

# 역전파 및 가중치 업데이트
loss.backward()
optimizer.step()

# 손실 축적
running_loss += loss.item()

# 에포크 끝날 때 평균 손실 출력
print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

# 테스트 데이터셋을 사용한 모델 평가
model.eval()

# 학습된 모델을 사용하여 결과물을 출력합니다.
fig, axes = plt.subplots(10, 2, figsize=(10, 20))

# 1에서 10까지의 이미지를 선택합니다.
for i, (input_image, _) in enumerate(test_loader):
    if i >= 10:
        break

    # 입력 이미지를 28x28 크기에서 784 크기로 펼칩니다.
    input_image = input_image.view(input_image.size(0), -1)

    # 학습된 모델을 사용하여 출력 이미지를 생성합니다.
    with torch.no_grad():
        output_image = model(input_image)

    # 입력 이미지를 시각화합니다.
    axes[i, 0].imshow(input_image.view(28, 28), cmap='gray')
    axes[i, 0].set_title(f'Original {i + 1}')
    axes[i, 0].axis('off')

    # 출력 이미지를 시각화합니다.
    axes[i, 1].imshow(output_image.view(28, 28), cmap='gray')
    axes[i, 1].set_title(f'Reconstructed {i + 1}')
    axes[i, 1].axis('off')

plt.tight_layout()
plt.show()
