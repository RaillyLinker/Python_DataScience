import torch
from torch import nn

"""
[게이트 순환 유닛(Gated Recurrent Unit, GRU)]
- GRU(Gated Recurrent Unit)는 2014년 뉴욕대학교 조경현 교수님이 집필한 논문에서 제안되었습니다. 
    GRU는 LSTM의 장기 의존성 문제에 대한 해결책을 유지하면서, 은닉 상태를 업데이트하는 계산을 줄였습니다. 
    다시 말해서, GRU는 성능은 LSTM 의 핵심 컨셉은 살리면서, 복잡했던 LSTM의 구조를 축약하였습니다.
    
- GRU 의 구조
    LSTM에서는 출력, 입력, 삭제 게이트라는 3개의 게이트가 존재했습니다. 
    반면, GRU에서는 업데이트 게이트와 리셋 게이트 두 가지 게이트만이 존재합니다.
    또한 LSTM 과는 달리 장기 메모리 역할을 하는 셀 스테이트가 사라졌고,
    히든 스테이트가 장기와 단기 메모리 역할을 겸한다고 합니다.
    
    아래의 구조 상세 설명은 인터넷에서 GRU 구조 도식을 보며 읽으시면 좋습니다.

- 리셋 게이트
    GRU 에서는 먼저 리셋 게이트가 실행됩니다.
    히든 스테이트와 x 값을 사용하여 리셋 게이트의 벨브 값을 만들고, 이를 히든 스테이트에 곱합니다.
    
- 업데이트 게이트
    역시나 x 값과 리셋 게이트에서 사용한 것과 같은 히든 스테이트를 가지고 게이트 벨브 값을 만들어냅니다.
    게이트 연산은 여기서 끝인데, 이 업데이트 게이트 값으로 다음에 넘겨줄 히든 스테이트를 만들어 주는 것입니다.

- 출력
    출력시에는 리셋 게이트로 얻어진 히든 스테이트 값과, x 를 사용하여 tanh 값을 얻고, 이 값에 업데이트 게이트로 나온
    벨브값의 역수를 곱해준 후, 갱신된 히든 스테이트를 더해주어 새로운 히든 스테이트 값과 출력값을 얻어냅니다.

- GRU와 LSTM 중 어떤 것이 모델의 성능면에서 더 낫다라고 단정지어 말할 수 없으며, 
    기존에 LSTM을 사용하면서 최적의 하이퍼파라미터를 찾아낸 상황이라면 굳이 GRU로 바꿔서 사용할 필요는 없습니다.
    경험적으로 데이터 양이 적을 때는 매개 변수의 양이 적은 GRU가 조금 더 낫고, 데이터 양이 더 많으면 LSTM이 더 낫다고도 합니다. 
    GRU보다 LSTM에 대한 연구나 사용량이 더 많은데, 이는 LSTM이 더 먼저 나온 구조이기 때문입니다.

- 아래 코드는 torch 에서 제공하는 GRU 레이어를 생성하여 순전파 하는 예시 입니다.
"""

input_size = 128
output_size = 256  # GRU에서는 hidden_size가 여기에 해당됩니다.
num_layers = 3
bidirectional = True

# LSTM 대신 GRU를 사용합니다.
model = nn.GRU(
    input_size=input_size,
    hidden_size=output_size,
    num_layers=num_layers,
    batch_first=True,
    bidirectional=bidirectional,
)

batch_size = 4
sequence_len = 6

inputs = torch.randn(batch_size, sequence_len, input_size)
# GRU는 c_0 상태를 사용하지 않으므로, h_0만 정의합니다.
h_0 = torch.rand(
    num_layers * (2 if bidirectional else 1),  # 양방향인 경우 2를 곱합니다.
    batch_size,
    output_size,
)

# GRU 모델에 inputs과 h_0을 넣어서 실행합니다. GRU는 LSTM과 달리 c_0을 반환하지 않습니다.
outputs, h_n = model(inputs, h_0)

print(outputs.shape)  # 출력값 형태
print(h_n.shape)  # 히든 스테이트 형태

# GRU는 LSTM과 다르게 셀 상태 c를 사용하지 않으므로 c_n에 대한 출력은 없습니다.
