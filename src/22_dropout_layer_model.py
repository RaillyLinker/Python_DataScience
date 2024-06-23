from torch import nn

"""
[드롭아웃 레이어]
- 드롭아웃은 과적합을 방지하기 위하여 학습시 일부 가중치를 비활성화 시키는 기법입니다.

- 드롭아웃 비율은 0.2에서 0.5 사이의 값을 사용하는 것이 일반적입니다.

- Dropout 레이어를 효과적으로 배치하는 위치
    입력 레이어 이후: 
        입력 데이터에 대한 초기 과적합을 방지하기 위해 입력 레이어 이후에 Dropout을 적용하는 것이 일반적입니다. 
        이는 입력 데이터의 특징을 다양하게 고려하여 모델이 더 강건하게 학습하도록 돕습니다.
    
    은닉 레이어 중간: 
        네트워크의 중간에 Dropout을 배치하여 모델의 복잡성을 줄이고 일반화 성능을 향상시킬 수 있습니다. 
        주로 모든 은닉 레이어 이후에 Dropout을 추가하는 것이 일반적이지만, 
        매우 깊거나 복잡한 모델의 경우 몇 개의 은닉 레이어만 선택적으로 Dropout을 적용할 수도 있습니다.
    
    출력 레이어 이전: 
        출력 레이어 이전에 Dropout을 적용하여 모델의 예측을 안정화시킬 수 있습니다. 
        특히 분류 문제에서는 출력 레이어 이전에 Dropout을 적용하여 클래스 간의 경계를 더욱 명확하게 만들 수 있습니다.
"""


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 10)
        self.dropout = nn.Dropout(p=0.5)
        self.layer2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x
