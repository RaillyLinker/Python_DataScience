import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

"""
[코사인 유사도(Cosine Similarity) 를 이용한 추천 시스템]
- 앞서 TF-IDF 를 통하여 단어를 빈도 기반의 중요도 벡터로 만들었습니다.
    빈도건, 중요도건, 속성이건 또 어떤 방식으로든 어떠한 일관된 기준으로 측정된 수치화된 데이터 벡터화가 가능하다면,
    각 벡터화된 데이터가 서로 얼마나 유사한지를 구할 수 있습니다.
    이렇게 유사도를 구할 수 있다면 가장 먼저 구현할 수 있는 가장 효율적인 서비스는 추천 시스템일 것입니다.
    유저가 특정 데이터를 자주 접하거나, 혹은 그에 긍정적 피드백을 보였다면,
    그와 유사한 정보를 노출시킴으로써 서비스의 질과 유저 경험을 개선할 수 있게 됩니다.

- 유사도를 측정할 수 있는 방법은 여러가지가 존재하는데, 여기서는 코사인 유사도를 사용하여 측정할 것입니다.
    코사인 유사도는 두 벡터 간의 코사인 각도를 이용하여 구할 수 있는 두 벡터의 유사도를 의미합니다.
    두 벡터의 방향이 완전히 동일한 경우에는 1의 값을 가지며,
    90°의 각을 이루면 0, 180°로 반대의 방향을 가지면 -1의 값을 갖게 됩니다.
    즉, 결국 코사인 유사도는 -1 이상 1 이하의 값을 가지며 값이 1에 가까울수록 유사도가 높다고 판단할 수 있습니다.
    이를 직관적으로 이해하면 두 벡터가 가리키는 방향이 얼마나 유사한가를 의미합니다.

- 수식으로 나타내면,
    (A * B) / (||A|| * ||B||)
    입니다.
    분자의 (A * B) 는 A 벡터와 B 벡터를 dot product 하는 것이고,
    분모의 (||A|| * ||B||) 은 A 의 norm 과 B 의 norm 을 곱하는 것으로,
    norm 은 이전에 loss regularization 에서 설명했기에 생략합니다.
"""


# [유사도 공식 구현]
def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))


# 테스트를 위한 예시 데이터 (각 문서에 대하여 TF-IDF 를 사용하여 아래와 같은 벡터가 추출되었다고 하겠습니다.)
doc1 = np.array([0, 1, 1, 1])
doc2 = np.array([1, 0, 1, 1])
doc3 = np.array([2, 0, 2, 2])

# 출력 : 문서 1과 문서1의 유사도 : 1.0000000000000002
print('문서 1과 문서1의 유사도 :', cos_sim(doc1, doc1))
# 출력 : 문서 1과 문서2의 유사도 : 0.6666666666666667
print('문서 1과 문서2의 유사도 :', cos_sim(doc1, doc2))
# 출력 : 문서 1과 문서3의 유사도 : 0.6666666666666667
print('문서 1과 문서3의 유사도 :', cos_sim(doc1, doc3))
# 출력 : 문서 2와 문서3의 유사도 : 1.0000000000000002
print('문서 2와 문서3의 유사도 :', cos_sim(doc2, doc3))
# 위 결과에서 동일한 문서에서 나온 벡터간의 유사도는 당연히 1 이 나오는데,
# 문서 2 와 문서 3 의 유사도 역시 1 입니다.
# 1 0 1 1 과 2 0 2 2 는 벡터의 방향이 완전히 동일하기에 위와 같은 결과가 나오는 것입니다.


# [유사도를 이용한 추천 시스템 구현하기]
# 캐글에서 사용되었던 영화 데이터셋을 가지고 영화 추천 시스템을 만들어보겠습니다.
# TF-IDF와 코사인 유사도만으로 영화의 줄거리에 기반해서 영화를 추천하는 추천 시스템을 만들 수 있습니다.
# (다운로드 링크 : https://www.kaggle.com/rounakbanik/the-movies-dataset)
# 원본 파일은 위 링크에서 movies_metadata.csv 파일을 다운로드 받으면 됩니다.
# 해당 데이터는 총 24개의 열을 가진 45,466개의 샘플로 구성된 영화 정보 데이터입니다.
data = pd.read_csv('../resources/datasets/33_cosine_similarity_recommend_system_code/movies_metadata.csv', low_memory=False)

# 상위 2만개의 샘플을 data에 저장
data = data.head(20000)

# overview 열에 존재하는 모든 결측값을 전부 카운트하여 출력
print('overview 열의 결측값의 수:', data['overview'].isnull().sum())

# 결측값을 빈 값으로 대체
data['overview'] = data['overview'].fillna('')

# 줄거리 데이터를 TF-IDF 벡터화
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['overview'])
print('TF-IDF 행렬의 크기(shape) :', tfidf_matrix.shape)

# 코사인 유사도 연산 (모든 줄거리 데이터의 벡터를 모든 줄거리 데이터 벡터와 연산한 행렬)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print('코사인 유사도 연산 결과 :', cosine_sim.shape)

# 영화 타이틀을 타이틀-인덱스 딕셔너리로 변경
title_to_index = dict(zip(data['title'], data.index))

# 영화 제목 Father of the Bride Part II의 인덱스를 리턴
idx = title_to_index['Father of the Bride Part II']
print(idx)


# 추천 목록 가져오기 = 입력받은 타이틀의 줄거리 벡터와 유사도가 높은 데이터들을 차례로 정렬하여 반환합니다.
def get_recommendations(title, cosine_sim=cosine_sim):
    # 선택한 영화의 타이틀로부터 해당 영화의 인덱스를 받아온다.
    idx = title_to_index[title]

    # 해당 영화와 모든 영화와의 유사도를 가져온다.
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 유사도에 따라 영화들을 정렬한다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 10개의 영화를 받아온다.
    sim_scores = sim_scores[1:11]

    # 가장 유사한 10개의 영화의 인덱스를 얻는다.
    movie_indices = [idx[0] for idx in sim_scores]

    # 가장 유사한 10개의 영화의 제목을 리턴한다.
    return data['title'].iloc[movie_indices]


print(get_recommendations('The Dark Knight Rises'))
