import torch
import torch.nn as nn
from gensim.models import Word2Vec
import numpy as np

"""
[파이토치(PyTorch)의 nn.Embedding()]
- 파이토치에서는 임베딩 벡터를 사용하는 방법이 크게 두 가지가 있습니다.
    바로 임베딩 층(embedding layer)을 만들어 훈련 데이터로부터 처음부터 임베딩 벡터를 학습하는 방법과
    미리 사전에 훈련된 임베딩 벡터(pre-trained word embedding)들을 가져와 사용하는 방법입니다.
    이번 챕터에서는 전자에 해당되는 방법에 대해서 배웁니다. 파이토치에서는 이를 nn.Embedding()를 사용하여 구현합니다.

- 임베딩 층은 룩업 테이블입니다.
    임베딩 모델은 학습이 필요한 신경망 모델이죠?
    그렇다면 학습이 완료된 시점에 단어를 임베딩 하려면 어떻게 해야할까요?
    
    원래라면 다시 신경망에 입력값을 입력하여 순전파 연산을 통하여 임베딩 벡터를 가져와야 할 것입니다.
    
    입력값으로는 학습시 사용한 것과 같이 단어를 one hot vector 로 바꿔서 넣어줘야겠네요.
    
    하지만 다름니다.
    
    임베딩 모델에서 모델 자체는 부산물입니다.
    중요한 것은 신경망의 파라미터가 아니라, 단어에 대응하는 결과값입니다.
    
    학습을 마친 임베딩 모델은 내부적으로 각 단어의 임베딩 벡터를 차곡차곡 쌓아둡니다.
    
    이렇게 되어, 학습을 완료한 임베딩 모델에 단어의 임베딩 벡터가 저장된 인덱스 번호를 입력하면, 
    이미 생성되어 저장되어 있던 임베딩 벡터를 가져와 사용할 수 있는 것입니다.
"""
# (vocabulary 와 embedding_table 구현)
train_data = 'you need to know how to code'
# 중복을 제거한 단어들의 집합인 단어 집합 생성.
word_set = set(train_data.split())
# 단어 집합의 각 단어에 고유한 정수 맵핑.
vocab = {word: i + 2 for i, word in enumerate(word_set)}
vocab['<unk>'] = 0
vocab['<pad>'] = 1
print(vocab)

# 임베딩 완료 된 결과 벡터들을 저장한 테이블이라고 생각하세요.
embedding_table = torch.FloatTensor(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.2, 0.9, 0.3],
        [0.1, 0.5, 0.7],
        [0.2, 0.1, 0.8],
        [0.4, 0.1, 0.1],
        [0.1, 0.8, 0.9],
        [0.6, 0.1, 0.1]
    ]
)

sample = 'you need to run'.split()
idxes = []
# 각 단어를 정수로 변환
for word in sample:
    try:
        idxes.append(vocab[word])
    # 단어 집합에 없는 단어일 경우 <unk>로 대체된다.
    except KeyError:
        idxes.append(vocab['<unk>'])
idxes = torch.LongTensor(idxes)

# 각 정수를 인덱스로 임베딩 테이블에서 값을 가져온다.
lookup_result = embedding_table[idxes, :]
print(lookup_result)

# (임베딩 층 사용하기)
train_data = 'you need to know how to code'

# 중복을 제거한 단어들의 집합인 단어 집합 생성.
word_set = set(train_data.split())

# 단어 집합의 각 단어에 고유한 정수 맵핑.
vocab = {tkn: i + 2 for i, tkn in enumerate(word_set)}
vocab['<unk>'] = 0
vocab['<pad>'] = 1

embedding_layer = nn.Embedding(num_embeddings=len(vocab),
                               embedding_dim=3,
                               padding_idx=1)

# train_data 의 단어 6 개와 <unk>, <pad> 의 임베딩 벡터로 하여 8 개의 벡터를 확인 할 수 있습니다.
print(embedding_layer.weight)

# 레이어 사용시에는 입력값으로 학습시 사용한 vocabulary 의 순서에 따른 단어의 인덱스 번호를 입력하면 됩니다.
# 그렇기에 기존 단어사전에 없는 단어를 넣으려면 <unk> 에 해당하는 인덱스 번호를 넣으면 됩니다.
print(embedding_layer(torch.tensor(3)))

"""
- 위에서 본 것처럼 torch 에서 제공하는 Embedding 모델을 사용하면 임베딩을 모델 안에 포함시켜 내부의 임베딩 벡터를 사용할 수 있습니다.
    그런데, 임베딩 모델, 임베딩 레이어에서 중요한 것은 모델 파라미터가 아닌 임베딩 벡터의 룩업 테이블이므로,
    이미 이미 학습된 다른 모델에서 이를 가져와 사용할 수 있습니다.
"""
# (저장된 Word2Vec 모델 불러오기)
model_file_path = "../_by_product_files/gensim_word_2_vec/word2vec.model"
word2vec = Word2Vec.load(model_file_path)

# 단어에서 인덱스로의 매핑 생성
word_to_index = {word: idx for idx, word in enumerate(word2vec.wv.index_to_key)}

# PyTorch의 nn.Embedding을 사용하여 임베딩 레이어 정의
embedding_dim = word2vec.vector_size
vocab_size = len(word_to_index)

embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# 임베딩 레이어를 Word2Vec 모델의 가중치로 초기화
embedding_layer.weight = nn.Parameter(torch.FloatTensor(word2vec.wv.vectors))

# 예시: 특정 단어에 대한 임베딩 얻기
word = '예시_단어'
word_index = word_to_index.get(word, word_to_index["<unk>"])  # 해당 단어가 어휘에 없으면 Unknown 인덱스
embedding = embedding_layer(torch.LongTensor([word_index]))

# 예시 단어에 대한 임베딩 출력
print(f"'{word}'에 대한 임베딩:\n{embedding}")

# ----------------------------------------------------------------------------------------------------------------------
# (저장된 Word2Vec 모델 불러오기 2)
word2vec = Word2Vec.load("../_by_product_files/gensim_word_2_vec/word2vec.model")
embedding_dim = word2vec.vector_size
id_to_token = word2vec.wv.index_to_key
vocab_size = len(id_to_token)
init_embeddings = np.zeros((vocab_size, embedding_dim))

for index, token in enumerate(id_to_token):
    init_embeddings[index] = word2vec.wv[token]

embedding_layer = nn.Embedding.from_pretrained(
    torch.tensor(init_embeddings, dtype=torch.float32)
)

# 예시: 특정 단어에 대한 임베딩 얻기
word = '예시_단어'
word_index = word_to_index.get(word, word_to_index["<unk>"])  # 해당 단어가 어휘에 없으면 Unknown 인덱스
embedding = embedding_layer(torch.LongTensor([word_index]))

# 예시 단어에 대한 임베딩 출력
print(f"'{word}'에 대한 임베딩:\n{embedding}")
