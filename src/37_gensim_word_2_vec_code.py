import pandas as pd
from Korpora import Korpora
from konlpy.tag import Okt
from gensim.models import Word2Vec
import os

"""
[Gensim Word 2 Vec]
gensim 라이브러리의 Word2Vec 은 계층적 소프트맥스나 네거티브 샘플링 등의 기법을 사용 하여 보다 쉽고 효과적으로 텍스트 임베딩 학습을 하도록 제공 해 줍니다.
또한 Cython 을 이용하여 C++ 기반 병렬처리 등으로 파이토치 임베딩 모델보다 더 빠른 속도를 보장합니다.
"""

# 학습할 네이버 영화 리뷰 데이터셋 말뭉치 로딩
corpus = Korpora.load("nsmc")

# 학습 데이터 프레임 추리기
corpus_train = pd.DataFrame(corpus.train)
if corpus_train.isnull().values.any():  # NULL 값 존재 유무
    corpus_train = corpus_train.dropna(how='any')  # 결측값이 존재하는 행을 제거

corpus_test = pd.DataFrame(corpus.test)
if corpus_test.isnull().values.any():  # NULL 값 존재 유무
    corpus_test = corpus_test.dropna(how='any')  # 결측값이 존재하는 행을 제거

# 한국어 형태소로 분리
tokenizer = Okt()
tokens = [
    # 단어 사전에 없을 모르는 단어는 <unk> 로 표현 하도록 이를 단어 사전에 포함 시킵니다.
    ["<unk>"],
    # 아무 의미 없이 형태를 맞추기 위한 패딩 토큰
    ["<pad>"]
]
for review in corpus_train.text:
    tokens.append(tokenizer.morphs(review))
for review in corpus_test.text:
    tokens.append(tokenizer.morphs(review))

print("tokens len :", len(tokens))  # 개수 출력

# 형태소 분리된 문장 리스트 3개 출력 :
# [
#   ['굳', 'ㅋ'],
#   ['GDNTOPCLASSINTHECLUB'],
#   ['뭐', '야', '이', '평점', '들', '은', '....', '나쁘진', '않지만', '10', '점', '짜', '리', '는', '더', '더욱', '아니잖아']
# ]
print(tokens[:3])

# gensim Word2Vec 모델 생성
word2vec = Word2Vec(
    sentences=tokens,
    # 임베딩 벡터 차원 수
    vector_size=128,
    # 학습 데이터 생성 윈도우 크기
    window=5,
    # 학습에 사용될 단어의 최소 빈도로, 이 값의 미만으로 나오는 단어는 학습에 사용하지 않습니다.
    min_count=1,
    # 학습시 사용할 스레드 개수
    workers=4,
    # Skip-gram 모델 사용 여부 (0 : CBoW 사용, 1 : Skip-Gram 사용)
    sg=1,
    # 학습 에폭 수
    epochs=3,
    # 단어 사전의 최대 크기 : 최소 빈도를 충족하는 단어가 최대 최종 단어 사전보다 많으면 자주 등장한 단어 순으로 단어 사전을 구축합니다.
    # max_final_vocab=50000
)

# 학습된 모델을 저장
model_file_save_directory_path = "../_by_product_files/gensim_word_2_vec"
if not os.path.exists(model_file_save_directory_path):
    os.makedirs(model_file_save_directory_path)
word2vec.save("../_by_product_files/gensim_word_2_vec/word2vec.model")

# 모델 파일에서 불러오기
word2vec = Word2Vec.load("../_by_product_files/gensim_word_2_vec/word2vec.model")

# 모델 테스트
word = "연기"
# 단어에 대한 임베딩 벡터 출력
print(word2vec.wv[word])
# 단어와 가장 비슷한 5 개의 단어 선정 및 유사도 확인
print(word2vec.wv.most_similar(word, topn=5))
# 두 단어간 유사도 확인
print(word2vec.wv.similarity(w1=word, w2="연기력"))

# OOV 테스트 (OOV 를 그냥 넣는다면 에러가 발생합니다.)
# KeyError: "Key '쀍궭휍' not present"
word_to_check = "쀍궭휍"
if word_to_check in word2vec.wv:
    embedding_vector = word2vec.wv[word_to_check]
else:
    embedding_vector = word2vec.wv["<unk>"]
    print(f"The word '{word_to_check}' is not present in the vocabulary. Using '<unk>' instead.")

print(embedding_vector)
