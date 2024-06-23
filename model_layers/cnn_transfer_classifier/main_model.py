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
        self.avgpool = vgg16.avgpool

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.classifier[0].weight)
        self.classifier[0].bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.classifier[3].weight)
        self.classifier[3].bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.classifier[6].weight)
        self.classifier[6].bias.data.fill_(0.01)

        # VGG 관련 레이어 파라미터는 학습하지 않게 동결
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.avgpool.parameters():
            param.requires_grad = False

    def forward(self, model_in):
        model_out = self.features(model_in)
        model_out = self.avgpool(model_out)
        model_out = torch.flatten(model_out, 1)  # Flatten them for FC
        model_out = self.classifier(model_out)
        return model_out
