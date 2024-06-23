import torch
from torch import nn
import torchvision.models as models


class MainModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 모델 내 레이어
        # VGG16 모델 불러오기 (pre-trained weights 사용)
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # VGG16의 첫 번째 conv layer부터 마지막 fully connected layer 전까지를 가져옴
        self.features = vgg16.features

        # GAP(Global Average Pooling) 레이어 -> FeatureMaps 의 각 Map 마다 평균을 구합니다.
        # 예를들어 (5,5) 크기의 맵이 6 개 있다고 하면, (5,5) 크기 맵의 평균 풀링을 하여 1 개의 값으로 추출하여, (6) 사이즈의 벡터로 만듭니다.
        # 이렇게 되면 Flatten 을 사용하지 않아도 최종 결정의 벡터가 나옵니다.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # GAP 에서 나온 벡터를 사용한 최종 분류용 은닉층
        self.classifier = nn.Linear(512, 10)

        self._init_weights()

    def _init_weights(self):
        # 마지막 fully connected layer의 weight 초기화
        nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0.01)

        # VGG 관련 레이어 파라미터는 학습하지 않게 동결
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, model_in):
        # Feature Maps 추출
        model_out = self.features(model_in)
        # 추출된 Feature Maps 저장
        # GAP(Global Average Pooling)
        model_out = self.avgpool(model_out)
        model_out = torch.squeeze(model_out)
        # 분류
        model_out = self.classifier(model_out)
        return model_out
