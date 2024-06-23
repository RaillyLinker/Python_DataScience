import model_layers.word_2_vec_skip_gram.main_model as word_2_vec_skip_gram
import os
from torch import nn
import pandas as pd
from Korpora import Korpora
from konlpy.tag import Okt
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch import optim
import numpy as np
from numpy.linalg import norm
import utils.torch_util as tu

"""
[Word2Vec]
- Word2Vec 은 2013 년 구글에서 공개한 임베딩 모델로 단어 간의 유사성을 측정하기 위해 분포 가설을 기반으로 개발된 신경망 기반의 텍스트 임베딩 방법입니다.

- 분포 가설(distributional hypothesis) 은 문맥상 동일한 위치에 존재하는 단어끼리는 서로 비슷한 의미를 지닌다는 것으로,
    내일 '자동차'를 타고 부산에 간다 와 내일 '비행기'를 타고 부산에 간다 라는 두 문장에서
    자동차와 비행기는 문맥상의 유사도와 탈것이라는 의미적 유사성을 지니고 있음을 알 수 있습니다.
    
    즉, 문장에서 특정 단어가 들어가는 문맥상 특정 부위를 비워둠으로써 해당 위치에 올 단어를 예측하는 모델을 학습시키면 해당 모델은 단어의 의미를 품고 있는
    투사층을 가지게 되는 것입니다. (학습시에는 예측 부분까지 사용하고, 의미 압축시에는 예측 부분을 제외한 투사층의 벡터가 의미 벡터 역할을 합니다.)
    위 예시 문장뿐 아니라 자동차와 비행기라는 단어가 들어가는 더 많고 다양한 예시들을 가지고 예측 모델을 학습시킨다면 탈것이라는 의미뿐 아니라
    더 상세한 정보가 담긴 투사층이 학습될 것입니다.
    
- CBOW(Continuous Bag of Words)
    Word2Vec에는 CBOW(Continuous Bag of Words)와 Skip-Gram 두 가지 방식이 있습니다. 
    CBOW는 주변에 있는 단어들을 가지고, 중간에 있는 단어들을 예측하는 방법입니다. 
    반대로, Skip-Gram은 중간에 있는 단어로 주변 단어들을 예측하는 방법입니다. 
    메커니즘 자체는 거의 동일하기 때문에 CBOW를 이해한다면 Skip-Gram도 손쉽게 이해 가능합니다. 
    우선 CBOW에 대해서 알아보도록 하겠습니다. 이해를 위해 매우 간소화 된 형태의 CBOW로 설명합니다.
    
    모델의 입력값은 모두 One-Hot-Vector 로 표현되는 상태입니다.

    예문 : "The fat cat sat on the mat"
    예를 들어서 갖고 있는 코퍼스에 위와 같은 문장이 있다고 합시다. 
    가운데 단어를 예측하는 것이 CBOW라고 했습니다. 
    {"The", "fat", "cat", "on", "the", "mat"}으로부터 sat을 예측하는 것은 CBOW가 하는 일입니다. 
    이 때 예측해야하는 단어 sat을 중심 단어(center word)라고 하고, 
    예측에 사용되는 단어들을 주변 단어(context word)라고 합니다.
    
    중심 단어를 예측하기 위해서 앞, 뒤로 몇 개의 단어를 볼지를 결정했다면 이 범위를 윈도우(window)라고 합니다. 
    예를 들어서 윈도우 크기가 2이고, 예측하고자 하는 중심 단어가 sat이라고 한다면 앞의 두 단어인 fat와 cat, 그리고 뒤의 두 단어인 on, the를 참고합니다. 
    윈도우 크기가 n이라고 한다면, 실제 중심 단어를 예측하기 위해 참고하려고 하는 주변 단어의 개수는 2n이 될 것입니다.
    
    윈도우 크기를 정했다면, 윈도우를 계속 움직여서 주변 단어와 중심 단어 선택을 바꿔가며 학습을 위한 데이터 셋을 만들 수 있는데, 
    이 방법을 슬라이딩 윈도우(sliding window)라고 합니다.

    위 예시에서, 윈도우 크기가 2 라고 하면, The 가 중심 단어일 때, 좌측에는 단어가 없으므로 fat, cat 이 주변단어고,
    한칸 우측으로 윈도우를 옮겨서 fat 이 중심단어 일 때, 좌측의 The, 우측의 cat, sat 이 주변 단어,
    한칸 더 우측으로 윈도우를 옮기면 cat 이 중심 단어, The, fat, sat, on 이 주변단어...
    
    이런식으로 입력값이 선정되며,
    
    cat 이 중심 단어일 때는 cat 이 모델이 예측해야할 정답이 되며,
    the, fat, sat, on 을 신경망에 입력하여 cat 이 반환되도록 모델을 학습하면 됩니다. 
    
    이때, 입력값은 The, fat, sat, on 의 각각의 one-hot-vector 의 리스트, 출력값은 score-vector 가 됩니다.
    
    출력층은 Softmax 로, 출력 결과가 각 단어에 속할 확률을 나타내는 벡터를 반환하는데, 이것이 바로 score-vector 로,
    결국 cat 의 one-hot-vector 와의 cross-entropy 값이 0 이 되는 방향으로 학습을 합니다.
    
- Skip-gram
    Skip-gram은 CBOW를 이해했다면, 메커니즘 자체는 동일하기 때문에 쉽게 이해할 수 있습니다. 
    앞서 CBOW에서는 주변 단어를 통해 중심 단어를 예측했다면, Skip-gram은 중심 단어에서 주변 단어를 예측하려고 합니다.
    
    입력값으로 cat 의 one-hot-vector 를 입력하고,
    출력값으로는, The, fat, sat, on 에 해당하는 4개의 score-vector 를 출력하는 것입니다.
    
    여러 논문에서 성능 비교를 진행했을 때, 전반적으로 Skip-Gram 이 CBoW 보다 성능이 좋다고 합니다.
    
- 아래 코드는 torch 모델을 가지고 기본적인 Skip-Gram 을 구현하는 방법을 정리합니다.
    물론 성능이 더 좋은 여러 기법을 적용한 라이브러리가 존재하며, 그 사용법은 다음 글에 정리하겠습니다.
"""


# [Torch 신경망으로 직접 Word2Vec Skip-Gram 구현하기]
def main():
    # 사용 가능 디바이스
    device = tu.get_gpu_support_device(gpu_support=True)

    # 데이터셋 객체 생성 (ex : tensor([[-10., 100., 82.], ...], device = cpu), tensor([[327.7900], ...], device = cpu))
    # 네이버 영화 리뷰 데이터셋 말뭉치 로딩
    corpus = Korpora.load("nsmc")

    # 학습 데이터 프레임 추리기
    corpus = pd.DataFrame(corpus.test)

    # 말뭉치 데이터를 한국어 형태소로 분리
    tokenizer = Okt()
    corpus_token_matrix = [tokenizer.morphs(review) for review in corpus.text]

    # 형태소 분리된 문장 리스트 3개 출력 :
    # [
    #   ['굳', 'ㅋ'],
    #   ['GDNTOPCLASSINTHECLUB'],
    #   ['뭐', '야', '이', '평점', '들', '은', '....', '나쁘진', '않지만', '10', '점', '짜', '리', '는', '더', '더욱', '아니잖아']
    # ]
    print(corpus_token_matrix[:3])

    # 형태소 분리된 문장 리스트들의 리스트로 단어사전 생성 함수
    # corpus_tokens 으로 단어사전 생성
    counter = Counter()
    for corpus_token_list in corpus_token_matrix:
        # tokens ex : ['굳', 'ㅋ']
        counter.update(corpus_token_list)
    # 사전에 없는 단어는 모두 special_token 으로 처리할 것
    vocab = ["<unk>"]
    for token, count in counter.most_common(5000):
        vocab.append(token)
    vocab = vocab

    # 단어사전 인덱스를 기반으로 토큰을 인덱스로 변경하는 dict
    vocab_token_to_id = {token: idx for idx, token in enumerate(vocab)}

    # 변경된 인덱스를 토큰으로 복구하는 dict
    id_to_vocab_token = {idx: token for idx, token in enumerate(vocab)}

    # 단어 사전 내 단어 10개 출력 :
    # ['<unk>', '.', '이', '영화', '의', '..', '가', '에', '...', '을']
    print(vocab[:10])
    # 단어사전 개수 출력 :
    # 5001 (n_vocab + special_token 개수)
    print(len(vocab))

    # tokens 를 skip-gram 모델 입력용 데이터로 가공
    # pair 리스트가 반환 되며, 첫번째는 중심 단어(입력값), 두번째는 생성 되어야 하는 주변 단어(정답값)가 됩니다.
    # window_size 는 중심 단어에 대해 출력 해야 할 단어를 주변 몇번째 까지로 할지에 대한 값 입니다.
    pairs = []
    window_size = 2
    for sentence in corpus_token_matrix:
        sentence_length = len(sentence)
        for idx, center_word in enumerate(sentence):
            window_start = max(0, idx - window_size)
            window_end = min(sentence_length, idx + window_size + 1)
            center_word = sentence[idx]
            context_words = sentence[window_start:idx] + sentence[idx + 1:window_end]
            for context_word in context_words:
                pairs.append([center_word, context_word])
    word_pairs = pairs
    # 출력 :
    # [
    #   ['굳', 'ㅋ'],
    #   ['ㅋ', '굳'],
    #   ['뭐', '야'],
    #   ['뭐', '이'],
    #   ['야', '뭐']
    # ]
    print(word_pairs[:5])

    # skip-gram 모델 입력용 데이터를 인덱스로 변경
    pairs = []
    unk_index = vocab_token_to_id["<unk>"]
    for word_pair in word_pairs:
        center_word, context_word = word_pair
        center_index = vocab_token_to_id.get(center_word, unk_index)
        context_index = vocab_token_to_id.get(context_word, unk_index)
        pairs.append([center_index, context_index])
    index_pairs = pairs
    # 인덱스 화 된 데이터 5 개 출력 :
    # [
    #   [595, 100],
    #   [100, 595],
    #   [77, 176],
    #   [77, 2],
    #   [176, 77]
    # ]
    print(index_pairs[:5])
    # 5001
    print(len(vocab))

    # 데이터셋 정리
    # (N, 2) 형태의 입력, 정답 pair 리스트를 텐서에 입력
    index_pairs = torch.tensor(index_pairs)

    # 중심 단어 리스트 분리
    center_indexs = index_pairs[:, 0]
    # tensor([595, 100,  77,  77, 176])
    print(center_indexs[:5])

    # 주변 단어 리스트 분리
    contenxt_indexs = index_pairs[:, 1]
    # tensor([100, 595, 176,   2,  77])
    print(contenxt_indexs[:5])

    # 중심 단어와 주변 단어 리스트로 데이터셋 준비 후 데이터 로더로 래핑
    dataset = TensorDataset(center_indexs, contenxt_indexs)
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 모델 생성
    model = word_2_vec_skip_gram.MainModel(
        vocab_size=len(vocab_token_to_id),
        embedding_dim=128
    )

    # 모델 학습
    tu.train_model(
        device=device,
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.SGD(model.parameters(), lr=0.1),
        train_dataloader=train_dataloader,
        num_epochs=10,
        # validation_dataloader=validation_dataloader,
        check_point_file_save_directory_path="../_check_point_files/word_2_vec_skip_gram",
        # check_point_load_file_full_path="../_check_point_files/~/checkpoint(2024_02_29_17_51_09_330).pt",
        log_freq=1
    )

    # 모델 저장
    model_file_save_directory_path = "../_torch_model_files/word_2_vec_skip_gram"
    if not os.path.exists(model_file_save_directory_path):
        os.makedirs(model_file_save_directory_path)
    save_file_full_path = tu.save_model_file(
        model=model,
        model_file_save_directory_path=model_file_save_directory_path
    )

    # # 저장된 모델 불러오기
    model = torch.load(save_file_full_path, map_location=device)
    print("Model Load Complete!")
    print(model)

    # 불러온 모델 테스트
    # 학습된 모델의 임베딩 계층에서 단어에 대한 임베딩 리스트의 리스트(매트릭스)를 가져와서 넘파이로 대입
    embedding_matrix = model.embedding.weight.detach().cpu().numpy()

    # 단어별 임베딩 벡터를 token_to_embedding 에 dict 타입(단어, 임베딩 리스트)으로 저장
    token_to_embedding = dict()
    for word, embedding in zip(vocab, embedding_matrix):
        token_to_embedding[word] = embedding

    # 단어사전 인덱스 30 번 단어의 임베딩 리스트 가져와 출력
    token = vocab[30]
    token_embedding = token_to_embedding[token]
    # ex : 연기
    print(token)
    # ex : [-0.8257925 1.3803437 ... 0.19158626 -0.74603665]
    print(token_embedding)

    # 임베딩 리스트는 단어의 의미를 지닌 벡터입니다.
    # 고로 서로 비슷한 단어는 서로 비슷한 벡터를 지닐 것입니다.
    # 이를 위하여 코사인 유사도를 측정하여 단어 임베딩 학습 결과를 확인할 것입니다.
    def cosine_similarity(a, b):
        cosine = np.dot(b, a) / (norm(b, axis=1) * norm(a))
        return cosine

    # 코사인 유사도에서 n 개를 추려오기
    def top_n_index(cosine_matrix, n):
        closest_indexes = cosine_matrix.argsort()[::-1]
        top_n = closest_indexes[1: n + 1]
        return top_n

    cosine_matrix = cosine_similarity(token_embedding, embedding_matrix)
    top_n = top_n_index(cosine_matrix, n=5)

    print(f"{token}와 가장 유사한 5 개 단어")
    for index in top_n:
        print(f"{id_to_vocab_token[index]} - 유사도 : {cosine_matrix[index]:.4f}")


if __name__ == '__main__':
    main()

"""
- Word2vec 모델은 자연어 처리에서 단어를 밀집 벡터로 만들어주는 단어 임베딩 방법론이지만 최근에 들어서는 자연어 처리를 넘어서 추천 시스템에도 사용되고 있는 모델입니다. 
    적당하게 데이터를 나열해주면 Word2vec은 위치가 근접한 데이터를 유사도가 높은 벡터를 만들어준다는 점에서 착안된 아이디어입니다. 
    구글에 'item2vec'을 검색해보세요.
"""
