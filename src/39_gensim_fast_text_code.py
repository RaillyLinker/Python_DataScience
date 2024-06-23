from Korpora import Korpora
from gensim.models import FastText
import os

"""
[Gensim fastText]
- 단어를 벡터로 만드는 또 다른 방법으로는 페이스북에서 개발한 FastText가 있습니다. 
    Word2Vec 이후에 나온 것이기 때문에, 메커니즘 자체는 Word2Vec의 확장이라고 볼 수 있습니다. 
    Word2Vec와 FastText와의 가장 큰 차이점이라면 Word2Vec는 단어를 쪼개질 수 없는 단위로 생각한다면, 
    FastText는 하나의 단어 안에도 여러 단어들이 존재하는 것으로 간주합니다. 
    내부 단어. 즉, 서브워드(subword)를 고려하여 학습합니다.
    
- 내부 단어(subword)의 학습
    FastText에서는 각 단어는 글자 단위 n-gram의 구성으로 취급합니다. 
    n을 몇으로 결정하는지에 따라서 단어들이 얼마나 분리되는지 결정됩니다. 
    예를 들어서 n을 3으로 잡은 트라이그램(tri-gram)의 경우, apple은 app, ppl, ple로 분리하고 이들을 벡터로 만듭니다. 
    더 정확히는 시작과 끝을 의미하는 <, >를 도입하여 아래의 5개 내부 단어(subword) 토큰을 벡터로 만듭니다.
    
    <ap, app, ppl, ple, le> 
    
    그리고 여기에 추가적으로 하나를 더 벡터화하는데, 기존 단어에 <, 와 >를 붙인 토큰입니다.
    
    <apple>
    
    다시 말해 n = 3인 경우, FastText는 단어 apple에 대해서 다음의 6개의 토큰을 벡터화하는 것입니다.
    
    <ap, app, ppl, ple, le>, <apple>
    
    그런데 실제 사용할 때는 n의 최소값과 최대값으로 범위를 설정할 수 있는데, 기본값으로는 각각 3과 6으로 설정되어져 있습니다. 
    다시 말해 최소값 = 3, 최대값 = 6인 경우라면, 단어 apple에 대해서 FastText는 아래 내부 단어들을 벡터화합니다.
    
    # n = 3 ~ 6인 경우
    <ap, app, ppl, ppl, le>, <app, appl, pple, ple>, <appl, pple>, ..., <apple>
    
    여기서 내부 단어들을 벡터화한다는 의미는 저 단어들에 대해서 Word2Vec을 수행한다는 의미입니다. 
    위와 같이 내부 단어들의 벡터값을 얻었다면, 단어 apple의 벡터값은 저 위 벡터값들의 총 합으로 구성합니다.
    
    apple = <ap + app + ppl + ppl + le> + <app + appl + pple + ple> + <appl + pple> + , ..., +<apple>
    
- 모르는 단어(Out Of Vocabulary, OOV)에 대한 대응
    FastText의 인공 신경망을 학습한 후에는 데이터 셋의 모든 단어의 각 n-gram에 대해서 워드 임베딩이 됩니다. 
    이렇게 되면 장점은 데이터 셋만 충분한다면 위와 같은 내부 단어(Subword)를 통해 모르는 단어(Out Of Vocabulary, OOV)에 대해서도 
    다른 단어와의 유사도를 계산할 수 있다는 점입니다.
    
    가령, FastText에서 birthplace(출생지)란 단어를 학습하지 않은 상태라고 해봅시다. 
    하지만 다른 단어에서 birth와 place라는 내부 단어가 있었다면, FastText는 birthplace의 벡터를 얻을 수 있습니다. 
    이는 모르는 단어에 제대로 대처할 수 없는 Word2Vec, GloVe와는 다른 점입니다.
    
- 단어 집합 내 빈도 수가 적었던 단어(Rare Word)에 대한 대응
    Word2Vec의 경우에는 등장 빈도 수가 적은 단어(rare word)에 대해서는 임베딩의 정확도가 높지 않다는 단점이 있었습니다. 
    참고할 수 있는 경우의 수가 적다보니 정확하게 임베딩이 되지 않는 경우입니다.

    하지만 FastText의 경우, 만약 단어가 희귀 단어라도, 그 단어의 n-gram이 다른 단어의 n-gram과 겹치는 경우라면, 
    Word2Vec과 비교하여 비교적 높은 임베딩 벡터값을 얻습니다.
    
    FastText가 노이즈가 많은 코퍼스에서 강점을 가진 것 또한 이와 같은 이유입니다. 
    모든 훈련 코퍼스에 오타(Typo)나 맞춤법이 틀린 단어가 없으면 이상적이겠지만, 실제 많은 비정형 데이터에는 오타가 섞여있습니다. 
    그리고 오타가 섞인 단어는 당연히 등장 빈도수가 매우 적으므로 일종의 희귀 단어가 됩니다. 
    즉, Word2Vec에서는 오타가 섞인 단어는 임베딩이 제대로 되지 않지만 FastText는 이에 대해서도 일정 수준의 성능을 보입니다.
    
    예를 들어 단어 apple과 오타로 p를 한 번 더 입력한 appple의 경우에는 실제로 많은 개수의 동일한 n-gram을 가질 것입니다.
"""

# 한국어 자연어 이해 데이터셋(Korean Natural Language Inference) 말뭉치 로딩
# kornli 는, 자연어 추론을 위한 데이터 셋으로, 자연어 추론이란, 두개 이상의 문장이 주어졌을 때, 두 문장 간의 관계를 분류하는 작업을 의미합니다.
corpus = Korpora.load("kornli")
corpus_texts = corpus.get_all_texts() + corpus.get_all_pairs()

# 텍스트 데이터를 공백으로 나누기
# fastText 는, 입력 단어의 구조적 특징을 학습할 수 있으므로, 형태소 분리를 할 필요가 없다고 합니다.
tokens = [sentence.split() for sentence in corpus_texts]

# 텍스트 데이터 문장 3개 출력
# [
#   ['개념적으로', '크림', '스키밍은', '제품과', '지리라는', '두', '가지', '기본', '차원을', '가지고', '있다.'],
#   ['시즌', '중에', '알고', '있는', '거', '알아?', '네', '레벨에서', '다음', '레벨로', '잃어버리는', '거야', '브레이브스가',
#       '모팀을', '떠올리기로', '결정하면', '브레이브스가', '트리플', 'A에서', '한', '남자를', '떠올리기로', '결정하면', '더블', 'A가',
#       '그를', '대신하러', '올라가고', 'A', '한', '명이', '그를', '대신하러', '올라간다.'
#   ],
#   ['우리', '번호', '중', '하나가', '당신의', '지시를', '세밀하게', '수행할', '것이다.']
# ]
print(tokens[:3])

# FastText 모델 생성
fastText = FastText(
    sentences=tokens,
    # 임베딩 벡터 차원 수
    vector_size=128,
    # 학습 데이터 생성 윈도우 크기
    window=5,
    # 학습에 사용될 단어의 최소 빈도로, 이 값의 미만으로 나오는 단어는 학습에 사용하지 않습니다.
    min_count=5,
    # Skip-gram 모델 사용 여부 (0 : CBoW 사용, 1 : Skip-Gram 사용)
    sg=1,
    # 학습시 사용할 스레드 개수
    workers=4,
    # 학습 에폭 수
    epochs=3,
    # N-Gram 최소값
    min_n=2,
    # N-Gram 최대값
    max_n=6
)

# 학습된 모델을 저장
model_file_save_directory_path = "../_by_product_files/gensim_fast_text"
if not os.path.exists(model_file_save_directory_path):
    os.makedirs(model_file_save_directory_path)
fastText.save("../_by_product_files/gensim_fast_text/fastText.model")

# 모델 파일에서 불러오기
fastText = FastText.load("../_by_product_files/gensim_fast_text/fastText.model")

# OOV 테스트
oov_token = "사랑해요"
oov_vector = fastText.wv[oov_token]
# 아래 출력에서 보다시피, 단어 "사랑해요" 는 단어 사전에 없습니다.
print(oov_token in fastText.wv.index_to_key)
# 하지만 유의미한 벡터를 얻을 수 있고, 그에 가장 비슷한 단어들(사랑, 사랑에, 사랑의, 등...)을 가져올 수 있습니다.
print(fastText.wv.most_similar(oov_vector, topn=5))
