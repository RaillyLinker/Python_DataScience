import torch
from torch import nn
import numpy as np

"""
[RNN]
- 순환 신경망(Recurrent Neural Network, RNN)
    RNN(Recurrent Neural Network)은 시퀀스(Sequence) 모델입니다. 
    입력과 출력을 시퀀스 단위로 처리하는 모델입니다. 
    번역기를 생각해보면 입력은 번역하고자 하는 문장. 즉, 단어 시퀀스입니다. 
    출력에 해당되는 번역된 문장 또한 단어 시퀀스입니다. 
    이러한 시퀀스들을 처리하기 위해 고안된 모델들을 시퀀스 모델이라고 합니다. 
    그 중에서도 RNN은 딥 러닝에 있어 가장 기본적인 시퀀스 모델입니다.
    
- 용어는 비슷하지만 순환 신경망(Recurrent Neural Network)과 재귀 신경망(Recursive Neural Network)은 전혀 다른 개념입니다.

- 앞서 배운 신경망들은 전부 은닉층에서 활성화 함수를 지난 값은 오직 출력층 방향으로만 향했습니다. 
    그런데 그렇지 않은 신경망들이 있습니다. 
    RNN(Recurrent Neural Network) 또한 그 중 하나입니다. 
    RNN은 은닉층의 노드에서 활성화 함수를 통해 나온 결과값을 출력층 방향으로도 보내면서, 
    다시 은닉층 노드의 다음 계산의 입력으로 보내는 특징을 갖고있습니다. (뱉은 출력을 다시 먹고 뱉기를 반복)
    이와 대조하여, 기존 신경망들을 피드 포워드 신경망(Feed Forward Neural Network)이라고 분류할 수 있습니다.

- RNN에서 은닉층에서 활성화 함수를 통해 결과를 내보내는 역할을 하는 노드를 셀(cell)이라고 합니다. 
    이 셀은 이전의 값을 기억하려고 하는 일종의 메모리 역할을 수행하므로 이를 메모리 셀 또는 RNN 셀이라고 표현합니다.
    은닉층의 메모리 셀은 각각의 시점(time step)에서 바로 이전 시점에서의 은닉층의 메모리 셀에서 나온 값을 자신의 입력으로 사용하는 재귀적 활동을 하고 있습니다. 
    이는 현재 시점 t 에서의 메모리 셀이 갖고 있는 값은 과거의 메모리 셀들의 값에 영향을 받은 것임을 의미합니다. 
    그렇다면 메모리 셀이 갖고 있는 이 값은 뭐라고 부를까요?
    메모리 셀이 출력층 방향으로 또는 다음 시점 t+1의 자신에게 보내는 값을 은닉 상태(hidden state)라고 합니다. 
    다시 말해 t 시점의 메모리 셀은 t-1 시점의 메모리 셀이 보낸 은닉 상태값을 t 시점의 은닉 상태 계산을 위한 입력값으로 사용합니다.
    
- RNN은 입력값이 순서를 지니는 시계열 데이터 분석에 큰 효용을 지닙니다.
    자연어가 바로 이러한 데이터이므로 한때, RNN 은 자연어 딥러닝의 주류로 사용되었죠.
    이전 데이터의 분석 결과가 있을 때 현재의 분석 결과가 달라지고, 현재의 분석 결과가 다음의 분석 결과에 영향을 끼치니,
    단어 하나하나의 입력 순서에 따라 전체적인 문장의 분석 결과를 다르게 판별하는 것입니다.
    예를들어, 이메일로 전송된 자연어 문구가 존재하면, 
    이를 토큰화하여 순서대로 입력하며 분석한 결과로 해당 문구가 스팸 메일인지 아닌지를 분류하는 방식으로 사용 할 수 있습니다.

- RNN 은 입력층으로 들어오는 토큰수에 상관 없이 동작합니다.
    즉, 한번의 예측에 순환되는 횟수가 매우 커질 수 있다는 것입니다.
    이렇게 되면 학습시 오차 역전파를 할 때, 기울기 소실 문제가 벌어지기 쉽다는 것을 알 수 있습니다.
    그렇기에 RNN 모델을 설계할 때에는, 
    활성화 함수로 시그모이드가 아닌 하이퍼볼릭 탄젠트(ReLU 를 사용하기도 합니다.)를 사용하거나, 
    배치 정규화를 사용하거나 하는 등...
    기울기 소실에 대비한 기법들을 동원해야만 할 것입니다.
    
- RNN 의 다른 한계점으로는, RNN 이 결국 바로 이전의 입력값을 먹은 출력값에 의존한다는 것입니다.
    무슨 말이냐면, 문장의 핵심이 되는 단어는 100 시퀀스 전에 이미 노출이 되었고, 그 중간에 부수적인 시퀀스로 차 있을 때,
    핵심 단어의 의미가 갈수록 옅어지는 것입니다.
    이는 마치 땅을 파서 보물찾기를 할 때, 보물이 이미 나왔지만, 
    그 위에 새로 퍼낸 흙을 덮고 또 덮어서, 최종적으로는 보물의 모습이 보이지 않게 되는 것과 같습니다.
    이러한 한계 때문에 추후 LSTM, 혹은 양방향 순환 신경망 같이 실질적인 보물을 재 환기하는 기법들이 개발되었는데,
    결국 현재 시점으로는 Attention 이라는 개념으로 보물과 그렇지 않은 것들에 점수를 매기는 형식을 사용하는 Transformer 가 나온 이후로
    이 문제가 해결이 되었습니다.
    결국 RNN 은 현재 자연어 처리의 주류가 아니라고 할 수 있지만, 그럼에도 추후 어떻게 사용될지는 모르는 일입니다.
    
- 양방향 순환 신경망(Bidirectional Recurrent Neural Network)
    RNN 의 구조는 순차적으로 데이터를 입력하고 압축하고 다시 입력하고를 반복하는 일입니다.
    당연히 순차적이므로 순차적인 데이터 처리에 적합하겠죠.
    그런데 한가지 예시를 보여드리겠습니다.
    
    Exercise is very effective at [          ] belly fat.
    
    위와 같은 빈칸 채우기 문제가 있을 때,
    RNN 으로 이를 해결한다고 합시다.
    
    at 다음에는 reducing 이라는 단어가 적할할 것입니다.
    
    그런데, 위 문장에서 Exercise is very effective at 까지만 보았을 때, 그것을 유추할 수 있을까요?
    이 경우 중요한 것은 뒤의 belly fat. 입니다.
    
    뒤에 무슨 시퀀스가 오느냐에 따라 정답이 달라지게 됩니다.
    
    고로 과거 시점(time step)의 데이터들을 사용하는 기본 RNN 모델은 위 빈칸 채우기 문제에 사용되기에는 효율적이지 못합니다.
    이에, 양방향으로 RNN 을 수행하는 양방향 RNN 이 나왔습니다.
    
    양방향 RNN은 하나의 출력값을 예측하기 위해 기본적으로 두 개의 메모리 셀을 사용합니다. 
    첫번째 메모리 셀은 앞에서 배운 것처럼 앞 시점의 은닉 상태(Forward States)를 전달받아 현재의 은닉 상태를 계산합니다. 
    두번째 메모리 셀은 앞에서 배운 것과는 다릅니다. 
    앞 시점의 은닉 상태가 아니라 뒤 시점의 은닉 상태(Backward States)를 전달 받아 현재의 은닉 상태를 계산합니다. 
    이 두 개의 값 모두가 출력층에서 출력값을 예측하기 위해 사용됩니다.
    
    양방향 RNN 의 구동 방식을 간단히 설명하자면,
    
    1. 입력 시퀀스를 받으며 순방향 순전파 실행
    2. 입력 시퀀스가 완료되면 역방향 순전파 실행
    3. 역방향 순전파가 완료되면 각 시점별 순방향 / 역방향 RNN State 가 각각 준비됨
    4. 각 시점별 결과값을 반환하려면 순방향 RNN State 와 역방향 RNN State 를 같이 사용하여 최종 출력을 함
    
    위와 같습니다.
    
"""
# (numpy 로 RNN 레이어 구현하기)
timesteps = 10  # 시점의 수. NLP에서는 보통 문장의 길이가 된다.
input_size = 4  # 입력의 차원. NLP에서는 보통 단어 벡터의 차원이 된다.
hidden_size = 8  # 은닉 상태의 크기이자, RNN 셀의 출력 사이즈이자, 메모리 셀의 용량이다.

inputs = np.random.random((timesteps, input_size))  # 입력에 해당되는 2D 텐서를 랜덤 생성
hidden_state_t = np.zeros((hidden_size,))  # 초기 은닉 상태는 0(벡터)로 초기화

Wx = np.random.random((hidden_size, input_size))  # (8, 4)크기의 2D 텐서 생성. 입력에 대한 가중치.
Wh = np.random.random((hidden_size, hidden_size))  # (8, 8)크기의 2D 텐서 생성. 은닉 상태에 대한 가중치.
b = np.random.random((hidden_size,))  # (8,)크기의 1D 텐서 생성. 이 값은 편향(bias).

# 시점별 히든 스테이트의 모음
total_hidden_states = []

# 메모리 셀 동작
for input_t in inputs:  # 각 시점에 따라서 입력값이 입력됨.
    # Wx * Xt + Wh * Ht-1 + b(bias)
    output_t = np.tanh(np.dot(Wx, input_t) + np.dot(Wh, hidden_state_t) + b)
    # 각 시점의 은닉 상태의 값을 계속해서 축적
    total_hidden_states.append(list(output_t))
    # 각 시점 t별 메모리 셀의 출력의 크기는 (timestep, output_dim)
    hidden_state_t = output_t

total_hidden_states = np.stack(total_hidden_states, axis=0)
# 출력 시 값을 깔끔하게 해준다.
print(total_hidden_states)  # (timesteps, output_dim)의 크기. 이 경우 (10, 8)의 크기를 가지는 메모리 셀의 2D 텐서를 출력.

# (torch 제공 RNN 레이어)
input_size = 128
hidden_size = 256
num_layers = 3
bidirectional = True

model = nn.RNN(
    # 입력 데이터 사이즈
    input_size=input_size,
    # RNN 히든 레이어 사이즈 = 출력 데이터 사이즈
    hidden_size=hidden_size,
    # 순환 신경망 층수
    num_layers=num_layers,
    # 활성화 함수 종류 (relu, tanh)
    nonlinearity="tanh",
    # 입력 배치 크기를 첫번째 차원으로 사용할지 여부
    # True : [배치 크기, 시퀀스 길이, 입력 특성 크기]
    # False : [시퀀스 길이, 배치 크기, 입력 특성 크기]
    batch_first=True,
    # 양방향 순환 여부
    bidirectional=bidirectional,
)

batch_size = 4
sequence_len = 6

# 테스트를 위한 무작위 입력값 생성
inputs = torch.randn(batch_size, sequence_len, input_size)

# RNN 에 입력될 초기 은닉 상태
h_0 = torch.rand(num_layers * (int(bidirectional) + 1), batch_size, hidden_size)

# 이번의 입력값과 기존 상태인 히든 값을 같이 넣기
outputs, hidden = model(inputs, h_0)

print(f"inputs shape : {inputs.shape}")
print(f"hidden shape : {hidden.shape}")
print(f"output shape : {outputs.shape}")
