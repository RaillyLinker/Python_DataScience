import numpy as np

"""
[벡터의 유사도 구하기]
- 벡터의 유사도를 구하기 위한 방법으로는 이전에 설명한 코사인 유사도 외에도 여러가지 방법들이 있습니다.
"""

"""
(코사인 유사도(Cosine Similarity))
- 전에 설명하였지만, 유사도를 구하는 방법을 정리하고 있으니, 코드없이 정리합니다.
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

"""
(유클리드 거리(Euclidean distance))
- 유클리드 거리, 또는 L2 거리는 가장 기본적인 거리 공식입니다.
    평면 좌표계에서 (x1, y1) 점과 (x2, y2) 점 좌표간 거리를 구할 때를 생각해봅시다.
    삼각형 빗변을 구하는 피타고라스 정리 공식에 따라,

    빗변의 길이 ** 2 = ((x2 - x1) ** 2) + (y2 - y1) ** 2

    이 됩니다.
    여기에 루트만 씌워준다면 거리가 구해지게 되죠.
    좌표간 거리가 바로 2차원 벡터의 거리입니다.

    차원 수를 늘려서, 다차원 공간에서 두개의 벡터 p 와 q가 각각 p = (p1, p2, p3, p4, ..., pn)과 q = (q1, q2, q3, q4, ..., qn) 라고 할 때,
    두 벡터 사이의 거리를 계산하는 유클리드 거리 공식은 다음과 같습니다.

    sqrt(sum((p - q)** 2))

- 위와 같이 간단하고 명료하지만, 유클리드 거리(euclidean distance)는 자연어 벡터의 유사도를 구할 때 자카드 유사도나 코사인 유사도만큼, 유용한 방법은 아닙니다.
"""


# (유클리드 거리 구현)
def dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


p = np.array([1, 2, 3])
q = np.array([4, 5, 6])

print('유클리드 거리 (L2) :', dist(p, q))

"""
(맨하탄 거리(Manhattan Distance))
- 맨하탄 거리, 또는 L1 거리는 두 점 사이의 거리를 계산할 때, 
    각 축에 대해 수평이나 수직으로만 이동할 수 있는 경우에 측정되는 거리입니다. 
    이 이름은 뉴욕의 맨하탄 지역이 격자 형태의 도로 구조를 가지고 있어, 한 지점에서 다른 지점으로 이동할 때 직선으로 이동할 수 없고, 
    도로의 격자를 따라 수평이나 수직으로만 이동할 수 있다는 점에서 유래되었습니다.

- p = (p1, p2, p3), q = (q1, q2, q3) 라는 벡터가 있을 때,
    d(p,q) = |p1 - q1| + |p2 - q2| + |p3 - q3|
    이런 식으로 구할 수 있습니다.
    
- 이 거리는 택시 기하학(Taxicab geometry)에서도 나타나며, 데이터 분석, 컴퓨터 비전, 경로 탐색 등 다양한 분야에서 활용됩니다.
"""


def manhattan_distance(point1, point2):
    """두 점(point1, point2) 사이의 맨하탄 거리를 계산합니다."""
    return sum(abs(a - b) for a, b in zip(point1, point2))


p = np.array([1, 2, 3])
q = np.array([4, 5, 6])
print("맨하탄 거리 (L1) :", manhattan_distance(p, q))

"""
[민코우스키 거리(Minkowski)]
- 민코우스키 거리는 두 점 사이의 거리를 측정하는 데 사용되는 일반화된 거리 공식입니다. 
    이 공식은 유클리드 거리와 맨해튼 거리를 포함하여 다양한 거리 측정법을 하나의 공식으로 통합합니다. 

- D(p,q) = (sum(|p-q|)**t) ** 1/t
    여기서,
    1. p 와 q 는 n 차원 공간에서 두 점을 나타냅니다.
    2. t 는 거리의 차수를 나타내며, t >= 1 입니다.
    
- t 값에 따라 민코우스키 거리는 다양한 거리 측정법으로 특수화될 수 있습니다:
    1. t = 1 일 때, 맨해튼 거리가 됩니다. (sum(|p-q|))
    2. t = 2 일 때, 유클리드 거리가 됩니다. (sum((|p-q|) ** 2)
"""


def minkowski_distance(p, q, p_value):
    return np.power(np.sum(np.abs(p - q) ** p_value), 1 / p_value)


# 예제 데이터 포인트
p = np.array([1, 2, 3])
q = np.array([4, 5, 6])

# 민코우스키 거리 계산
distance_l1 = minkowski_distance(p, q, 1)  # 맨해튼 거리
distance_l2 = minkowski_distance(p, q, 2)  # 유클리드 거리

print("민코우스키 거리 (L2) :", distance_l2)
print("민코우스키 거리 (L1) :", distance_l1)

"""
[채비쇼프 거리(Chaebichev Distance)]
- 채비쇼프 거리는 두 점 간의 거리를 측정하는 방법 중 하나로, 또는 L∞ 거리(infinity norm distance)로도 불립니다. 
    이 거리 측정 방법은 두 벡터(또는 점) 간의 각 차원별 차이 중에서 가장 큰 값으로 거리를 정의합니다.
    
- max(p - q) 으로 간단히 구할 수 있습니다.
"""


def chebyshev_distance(p, q):
    distance = np.max(np.abs(p - q))
    return distance


# 예제 데이터 포인트
p = np.array([1, 2, 3])
q = np.array([4, 5, 6])

# 채비쇼프 거리 계산
distance = chebyshev_distance(p, q)
print("채비쇼프 거리 :", distance)

"""
(자카드 유사도(Jaccard similarity))
- A와 B 두개의 집합이 있다고 합시다. 
    이때 교집합은 두 개의 집합에서 공통으로 가지고 있는 원소들의 집합을 말합니다. 
    즉, 합집합에서 교집합의 비율을 구한다면 두 집합 A와 B의 유사도를 구할 수 있다는 것이 자카드 유사도(jaccard similarity)의 아이디어입니다. 
    자카드 유사도는 0과 1사이의 값을 가지며, 만약 두 집합이 동일하다면 1의 값을 가지고, 두 집합의 공통 원소가 없다면 0의 값을 갖습니다.

- 정리하자면, (A, B 교집합) / (A, B 합집합) = (A, B 교집합) / (A 집합 + B 집합 - (A, B 교집합))
"""
# 자카드 유사도 구현
doc1 = "apple banana everyone like likey watch card holder"
doc2 = "apple banana coupon passport love you"

# 토큰화
tokenized_doc1 = doc1.split()
tokenized_doc2 = doc2.split()

# 출력 : 문서1 : ['apple', 'banana', 'everyone', 'like', 'likey', 'watch', 'card', 'holder']
print('문서1 :', tokenized_doc1)
# 출력 : 문서2 : ['apple', 'banana', 'coupon', 'passport', 'love', 'you']
print('문서2 :', tokenized_doc2)

union = set(tokenized_doc1).union(set(tokenized_doc2))
# 출력 :
# 문서1과 문서2의 합집합 :
# {'likey', 'watch', 'like', 'passport', 'card', 'holder', 'love', 'banana', 'coupon', 'apple', 'everyone', 'you'}
print('문서1과 문서2의 합집합 :', union)

intersection = set(tokenized_doc1).intersection(set(tokenized_doc2))
# 출력 : 문서1과 문서2의 교집합 : {'banana', 'apple'}
print('문서1과 문서2의 교집합 :', intersection)

# 출력 : 자카드 유사도 : 0.16666666666666666
print('자카드 유사도 :', len(intersection) / len(union))

"""
[마할라노비스 거리(Mahalanobis Distance)]
- 마할라노비스 거리는 다변량 데이터에서 두 점 간의 거리를 측정하는 방법 중 하나입니다. 
    이 거리는 각 차원 간의 상관 관계를 고려하여 데이터의 분산을 반영합니다. 
    데이터가 다변량 정규 분포를 따른다고 가정할 때 유용하게 사용됩니다.
"""


def mahalanobis_distance(p, q, covariance_matrix):
    diff = p - q
    inv_covariance_matrix = np.linalg.inv(covariance_matrix)
    distance = np.sqrt(np.dot(np.dot(diff.T, inv_covariance_matrix), diff))
    return distance


# 예제 데이터 포인트
p = np.array([1, 2, 3])
q = np.array([4, 5, 6])

# 예제 공분산 행렬 (가상의 값)
covariance_matrix = np.array([[2, 1, 0], [1, 2, 0], [0, 0, 1]])

# Mahalanobis 거리 계산
distance = mahalanobis_distance(p, q, covariance_matrix)
print("마할라노비스 거리 :", distance)

"""
[표준화 거리(Standardized Distance)]
- 표준화 거리는 거리 공식에서 각 벡터 값에 표준화를 적용한 것을 의미합니다.
    예를들어 유클리드 거리 공식인
    sqrt(sum((p - q)** 2))
    가 있다면, 여기에 표준화를 적용한다고 하면,
    여기서 전체 데이터의 평균과 표준편차를 사용하여,
    (p - 평균 / 표준편차)
    (q - 평균 / 표준편차)
    를 하여 각 데이터를 동일 기준으로 표준화 시켜서 그 거리를 구하는 것입니다.
    
- 표준화 거리는 다차원 공간에서 두 점 사이의 거리를 측정할 때 각 차원의 변동성을 고려하여 거리를 조정하는 방법입니다. 
    이는 각 변수(차원)가 서로 다른 스케일을 가질 때 유용하게 사용됩니다. 
    예를 들어, 한 변수의 단위가 미터이고 다른 변수의 단위가 킬로그램일 경우, 단순히 차이를 합산하는 것은 적절하지 않습니다. 
    이런 경우에 표준화 거리를 사용하여 각 차원을 표준화(예: 평균을 0, 표준편차를 1로 조정)한 뒤 거리를 계산합니다.

- 표준화 거리는 유클리드 거리뿐 아니라 다른 거리 공식에도 사용 가능합니다.
"""


def calculate_mean_std(data):
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    return means, stds


def standardized_euclidean_distance(p, q, means, stds):
    normalized_p = (p - means) / stds
    normalized_q = (q - means) / stds
    distance = np.sqrt(np.sum((normalized_p - normalized_q) ** 2))
    return distance


# 예제 데이터 포인트 집합
data = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# 평균과 표준편차 계산
means, stds = calculate_mean_std(data)

# 두 점 선택 (예제에서는 처음 두 점을 사용)
p = data[0]
q = data[1]

# 표준화 거리 계산
distance = standardized_euclidean_distance(p, q, means, stds)
print("표준화 유클리드 거리 :", distance)
