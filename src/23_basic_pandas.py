import pandas as pd
import numpy as np

"""
[Pandas]
- 판다스는 데이터 분석과 관련된 다양한 기능을 제공하는 파이썬 라이브러리입니다.
    데이터를 쉽게 조작하고 다룰 수 있도록 도와주며, 데이터 파악 및 전처리에 도움을 주기 때문에 딥러닝, 머신러닝을 공부하면 반드시 함께 배우게 됩니다.
    아래는 판다스의 대표적인 용법을 정리한 코드입니다.
"""

# (데이터 프레임 다루기)
# 데이터 프레임 만들기
# 데이터 프레임은 판다스의 기본적인 데이터 형식입니다.
# 데이터베이스의 테이블 행렬로 생각하면 편하며, 객체를 생성하는 것만으로 데이터 처리의 다양한 메소드를 제공합니다.
df = pd.DataFrame(
    # 데이터 행렬을 저장합니다.
    # dict 타입으로, key 는 컬럼명, 안의 List 는 Rows 의 데이터를 의미합니다.
    {
        "a": [4, 5, 6, 7],
        "b": [8, 9, 10, 11],
        "c": [12, 13, 14, 15]
    },
    # index 란, 행렬의 행의 이름을 의미합니다. (이것이 None 이라면 0부터 시작되는 인덱스이고, 데이터의 행 개수와 맞지 않는다면 에러가 납니다.)
    index=[1, 2, 3, 4]
)

# 데이터 프레임 출력
print(f"1 :\n{df}")

# 데이터의 열 이름을 따로 지정해서 데이터 프레임을 만들수도 있습니다.
# 실제 데이터베이스 테이블 형태와 동일해서 더 가독성이 좋을 수도 있습니다.
df = pd.DataFrame(
    [
        [4, 8, 12],
        [5, 9, 13],
        [6, 10, 14],
        [7, 11, 15]
    ],
    index=[1, 2, 3, 4],
    columns=['a', 'b', 'c']
)

print(f"\n2 :\n{df}")

# 인덱스를 두개로 설정할 수도 있습니다.
df = pd.DataFrame(
    [
        [4, 8, 12],
        [5, 9, 13],
        [6, 10, 14],
        [7, 11, 15]
    ],
    # 아래는 인덱스를 튜플 형태로 지정하여 2개를 한 세트로 설정하는 것입니다.
    # d 와 e 는 서로 같은 것이고, names 로 동일한 d 는 n 으로 묶고, e 는 v 로 묶은 것입니다.
    # 일종의 그루핑이라 생각하면 됩니다.
    index=pd.MultiIndex.from_tuples(
        [('d', 1), ('d', 2), ('e', 3), ('e', 4)],
        names=['n', 'v']
    ),
    columns=['a', 'b', 'c']
)

print(f"\n3 :\n{df}")

# (데이터 정렬하기)
# 특정 열 값을 기준으로 정렬하기
rdf = df.sort_values('a', ascending=False)
print(f"\n4 :\n{rdf}")

# 열 이름 변경하기 (c 열을 d 로 변경)
rdf = df.rename(columns={'c': 'd'})
print(f"\n5 :\n{rdf}")

# 인덱스 값 초기화 (인덱스가 0부터 시작하는 숫자로 초기화되고, 기존 인덱스는 컬럼으로 분리됩니다.)
rdf = df.reset_index()
print(f"\n6 :\n{rdf}")

# 인덱스 정렬
rdf = df.sort_index(ascending=False)
print(f"\n7 :\n{rdf}")

# 특정 열 제거
rdf = df.drop(columns=['a', 'b'])
print(f"\n8 :\n{rdf}")

# (행 추출하기)
# 맨 위의 행 추출
rdf = df.head(2)
print(f"\n9 :\n{rdf}")

# 맨 아래의 행 추출
rdf = df.tail(2)
print(f"\n10 :\n{rdf}")

# 특정 열의 값을 추출
rdf = df[df['a'] > 4]  # a 열 중 4보다 큰 값이 있을 경우 해당 행을 추출
print(f"\n11 :\n{rdf}")

# 특정 열에 특정 값이 있을 경우 추출
rdf = df[df['a'] == 6]  # a 열 중 6 이 있을 경우 해당 행을 추출
print(f"\n12 :\n{rdf}")

# 특정 열에 특정 값이 없을 경우 추출
rdf = df[df['a'] != 5]  # a 열 중 5 가 없을 경우 해당 행을 추출
print(f"\n13 :\n{rdf}")

# 특정 열에 특정 값들중 하나가 있을 경우 추출
rdf = df[df['a'].isin([4, 6])]
print(f"\n14 :\n{rdf}")

# 특정 비율로 추출
rdf = df.sample(frac=0.75)  # 75% 비율로 랜덤하게 행을 추출
print(f"\n15 :\n{rdf}")

# 특정 개수만큼 추출
rdf = df.sample(n=3)  # 3개를 랜덤하게 행을 추출
print(f"\n16 :\n{rdf}")

# 특정 열에서 큰 데이터를 큰 순서대로 n 개 추출
rdf = df.nlargest(3, 'a')
print(f"\n17 :\n{rdf}")

# 특정 열에서 작은 데이터를 작은 순서대로 n 개 추출
rdf = df.nsmallest(3, 'a')
print(f"\n18 :\n{rdf}")

# (열 추출하기)
# 인덱스의 범위로 불러오기 (0부터 시작하는 기본 인덱스가 기준입니다.)
rdf = df.iloc[1:3]  # 1, 2 번 인덱스 추출
print(f"\n19 :\n{rdf}")

# 특정 인덱스부터 끝까지
rdf = df.iloc[2:]
print(f"\n20 :\n{rdf}")

# 첫 인덱스부터 특정 인덱스 까지
rdf = df.iloc[:2]
print(f"\n21 :\n{rdf}")

# 모든 인덱스
rdf = df.iloc[:]
print(f"\n22 :\n{rdf}")

# 특정 열을 지정해 가져오기
rdf = df[['a', 'b']]
print(f"\n23 :\n{rdf}")

# 특정 문자가 포함된 열 가져오기
rdf = df.filter(regex='c')  # 열 이름에 c 가 포함되어 있다면 출력
print(f"\n24 :\n{rdf}")

# 특정 문자가 포함되지 않은 열 가져오기
rdf = df.filter(regex='^(?!c$).*')  # 열 이름에 c 가 포함되어 있지 않다면 출력
print(f"\n25 :\n{rdf}")

# (행과 열 추출하기)
# 특정 행과 열을 지정해서 가져오기
rdf = df.loc[:, 'a': 'c']  # 모든 인덱스에서, a 열부터 c 열까지 추출
print(f"\n26 :\n{rdf}")

# 인덱스로 특정 행과 열 가져오기
rdf = df.iloc[0:3, [0, 2]]  # 0 인덱스부터 2 인덱스 까지 0번째 열과 2번째 열을 추출
print(f"\n27 :\n{rdf}")

# 특정 열에서 조건을 만족하는 행과 열 가져오기
rdf = df.loc[df['a'] > 5, ['a', 'c']]  # a 열 값이 5보다 큰 행에서, a, c 열을 출력
print(f"\n28 :\n{rdf}")

# 인덱스를 이용해 특정 조건을 만족하는 값 불러오기
rdf = df.iat[1, 2]  # 1번째 인덱스, 2번째 열 값을 가져오기
print(f"\n29 :\n{rdf}")

# (중복 데이터 다루기)
# 중복값이 포함된 Dataframe 준비
df = pd.DataFrame(
    {
        "a": [4, 5, 6, 7, 7],
        "b": [8, 9, 10, 11, 11],
        "c": [12, 13, 14, 15, 15]
    },
    index=pd.MultiIndex.from_tuples(
        [('d', 1), ('d', 2), ('e', 1), ('e', 2), ('e', 3)],
        names=['n', 'v']
    ),
)
print(f"\n30 :\n{df}")

# 특정 열에 어떤 값이 몇개 들어있는지 확인
rdf = df['a'].value_counts()
print(f"\n31 :\n{rdf}")

# 데이터 프레임 행 개수
rdf = len(df)
print(f"\n32 :\n{rdf}")

# 데이터 프레임 행 / 열 개수
rdf = df.shape
print(f"\n33 :\n{rdf}")

# 특정 열에 유니크 값 확인
rdf = df['a'].unique()
print(f"\n34 :\n{rdf}")

# 데이터 프레임 형태 한눈에 확인
rdf = df.describe()
print(f"\n35 :\n{rdf}")

# 중복 값 제거
df = df.drop_duplicates()
print(f"\n36 :\n{df}")

# (데이터 파악하기)
# 각 열의 합 보기
rdf = df.sum()
print(f"\n37 :\n{rdf}")

# 각 열의 개수 확인
rdf = df.count()
print(f"\n38 :\n{rdf}")

# 각 열의 중간값 확인
rdf = df.median()
print(f"\n39 :\n{rdf}")

# 특정 열의 평균값 확인
rdf = df['b'].mean()
print(f"\n40 :\n{rdf}")

# 각 열의 25%, 75% 에 해당하는 수 보기
rdf = df.quantile([0.25, 0.75])
print(f"\n41 :\n{rdf}")

# 각 열의 최소값 보기
rdf = df.min()
print(f"\n42 :\n{rdf}")

# 각 열의 최대값 보기
rdf = df.max()
print(f"\n43 :\n{rdf}")

# 각 열의 표준편차 보기
rdf = df.std()
print(f"\n44 :\n{rdf}")

# 데이터프레임 내 모든 값에 일괄적으로 함수 적용하기
rdf = df.apply(np.sqrt)  # 각 값의 제곱근 구하기
print(f"\n45 :\n{rdf}")

# (결측치 다루기)
# 테스트를 위하여 결측치가 들어있는 데이터 프레임 생성
df = pd.DataFrame(
    {
        "a": [4, 5, 6, np.nan],
        "b": [7, 8, np.nan, 9],
        "c": [10, np.nan, 11, 12]
    },
    index=pd.MultiIndex.from_tuples(
        [('d', 1), ('d', 2), ('e', 1), ('e', 2)],
        names=['n', 'v']
    ),
)
print(f"\n46 :\n{df}")

# 결측치인지 확인
rdf = pd.isnull(df)
print(f"\n47 :\n{rdf}")

# 결측치가 아닌지 확인
rdf = pd.notnull(df)
print(f"\n48 :\n{rdf}")

# 결측치가 있는 행 삭제
rdf = df.dropna()
print(f"\n49 :\n{rdf}")

# 결측치를 특정 값으로 대체
rdf = df.fillna(13)
print(f"\n50 :\n{rdf}")

# 결측치를 특정 계산 결과로 대체
rdf = df.fillna(df['a'].mean())
print(f"\n51 :\n{rdf}")

# (새로운 열 만들기)
# 테스트를 위하여 데이터 프레임 생성
df = pd.DataFrame(
    {
        "a": [4, 5, 6, 7],
        "b": [8, 9, 10, 11],
        "c": [12, 13, 14, 15]
    },
    index=pd.MultiIndex.from_tuples(
        [('d', 1), ('d', 2), ('e', 1), ('e', 2)],
        names=['n', 'v']
    ),
)
print(f"\n52 :\n{df}")

# 조건에 맞는 새 열 만들기
df['sum'] = df['a'] + df['b'] + df['c']
print(f"\n53 :\n{df}")

# assign 함수를 사용하여 조건에 맞는 새 열 만들기
df = df.assign(multiply=lambda df: df['a'] * df['b'] * df['c'])
print(f"\n54 :\n{df}")

# 숫자형 데이터를 구간으로 나누기
# a 열을 2개로 나누어 각각 새롭게 레이블을 만들기
df['qcut'] = pd.qcut(df['a'], 2, labels=["a1", "a2"])
print(f"\n55 :\n{df}")

# 기준 값 이하와 이상을 모두 통일시키기
# a 열의 값 중 5 이하는 모두 5, 6 이상은 모두 6으로 만들기
df['clip'] = df['a'].clip(lower=5, upper=6)
print(f"\n56 :\n{df}")

# 최대값 불러오기
# axis=0 은 행과 행 비교, axis=1 은 열과 열 비교
rdf = df.max(axis=0)
print(f"\n57 :\n{rdf}")

# 최소값 불러오기
rdf = df.min(axis=0)
print(f"\n58 :\n{rdf}")

# (행과 열 변환하기)
# 테스트를 위하여 데이터 프레임 생성
df = pd.DataFrame(
    {
        "A": ['a', 'b', 'c'],
        "B": [1, 3, 5],
        "C": [2, 4, 6]
    },
)
print(f"\n59 :\n{df}")

# 모든 열을 행으로 변환하기
rdf = pd.melt(df)
print(f"\n60 :\n{rdf}")

# 하나의 열만 행으로 이동시키기
rdf = pd.melt(df, id_vars=['A'], value_vars=['B'])  # A열만 그대로, B열은 행으로 이동
print(f"\n61 :\n{rdf}")

# 여러개의 열을 행으로 이동시키기
rdf = pd.melt(df, id_vars=['A'], value_vars=['B', 'C'])  # A열만 그대로, B열, C열은 행으로 이동
print(f"\n62 :\n{rdf}")

# 특정 열의 값을 기준으로 새로운 열 만들기
rdf = rdf.pivot(index='A', columns='variable', values='value')
print(f"\n63 :\n{rdf}")

# 원래 데이터 형태로 되돌리기
rdf = rdf.reset_index()
rdf.columns.names = [None]
print(f"\n64 :\n{rdf}")

# (시리즈 데이터 연결하기)
# Dataframe 이 아니라 시리즈 객체 만들기 (컬럼 하나와 동일)
s1 = pd.Series(['a', 'b'])
print(f"\n65 :\n{s1}")

s2 = pd.Series(['c', 'd'])
print(f"\n66 :\n{s2}")

# 시리즈 합치기
rst = pd.concat([s1, s2])
print(f"\n67 :\n{rst}")

# 시리즈 합칠 때 기존 인덱스를 지우고 새로운 인덱스 만들기
rst = pd.concat([s1, s2], ignore_index=True)
print(f"\n68 :\n{rst}")

# 계층적 인덱스를 추가하고 열 이름 지정하기
rst = pd.concat([s1, s2], keys=['s1', 's2'], names=['Series name', 'Row ID'])
print(f"\n69 :\n{rst}")

# (데이터 프레임 연결하기)
# 테스트를 위하여 데이터 프레임 생성
df1 = pd.DataFrame(
    {
        "letter": ['a', 'b'],
        "number": [1, 2]
    },
)
print(f"\n70 :\n{df1}")

df2 = pd.DataFrame(
    {
        "letter": ['c', 'd'],
        "number": [3, 4]
    },
)
print(f"\n71 :\n{df2}")

df3 = pd.DataFrame(
    {
        "letter": ['c', 'd'],
        "number": [3, 4],
        "animal": ['cat', 'dog']
    },
)
print(f"\n72 :\n{df3}")

df4 = pd.DataFrame(
    {
        "animal": ['bird', 'monkey'],
        "name": ['polly', 'george']
    },
)
print(f"\n73 :\n{df4}")

# 데이터 프레임 합치기
rst = pd.concat([df1, df2])
print(f"\n74 :\n{rst}")

# 열의 수가 다른 두 데이터 프레임 합치기
rst = pd.concat([df1, df3])
print(f"\n75 :\n{rst}")

# 열의 수가 다른 두 데이터 프레임 inner join 합치기
rst = pd.concat([df1, df3], join='inner')
print(f"\n76 :\n{rst}")

# 열 이름이 서로 다른 데이터 합치기
rst = pd.concat([df1, df4], axis=1)
print(f"\n77 :\n{rst}")

# (데이터 병합하기)
# 테스트를 위하여 데이터 프레임 생성
df1 = pd.DataFrame(
    {
        "x1": ['A', 'B', 'C'],
        "x2": [1, 2, 3]
    },
)
print(f"\n78 :\n{df1}")

df2 = pd.DataFrame(
    {
        "x1": ['A', 'B', 'D'],
        "x3": ['T', 'F', 'T']
    },
)
print(f"\n79 :\n{df2}")

df3 = pd.DataFrame(
    {
        "x1": ['B', 'C', 'D'],
        "x2": [2, 3, 4]
    },
)
print(f"\n80 :\n{df3}")

# 왼쪽 열을 축으로 병합하기
# x1 을 키로 해서 병합, 왼쪽(adf)을 기준으로 왼쪽의 adf 에는 D가 없으므로 해당 값은 NaN으로 변환합니다.
rst = pd.merge(df1, df2, how='left', on='x1')
print(f"\n81 :\n{rst}")

# 오른쪽 열을 축으로 병합하기
rst = pd.merge(df1, df2, how='right', on='x1')
print(f"\n82 :\n{rst}")

# 공통 값만 병합하기
rst = pd.merge(df1, df2, how='inner', on='x1')
print(f"\n83 :\n{rst}")

# 모든 값을 병합하기
rst = pd.merge(df1, df2, how='outer', on='x1')
print(f"\n84 :\n{rst}")

# 특정한 열을 비교해서 공통 값이 존재하는 경우만 가져오기
rst = df1[df1.x1.isin(df2.x1)]
print(f"\n85 :\n{rst}")

# 공통 값이 존재하는 경우 해당 값을 제외하고 병합하기
rst = df1[~df1.x1.isin(df2.x1)]
print(f"\n86 :\n{rst}")

# 공통 값이 있는 것만 병합하기
rst = pd.merge(df1, df3)
print(f"\n87 :\n{rst}")

# 모두 병합하기
rst = pd.merge(df1, df3, how='outer')
print(f"\n88 :\n{rst}")

# 어디서 병합되었는지 표시하기
rst = pd.merge(df1, df3, how='outer', indicator=True)
print(f"\n89 :\n{rst}")

# 원하는 병합만 남기기
rst = pd.merge(df1, df3, how='outer', indicator=True).query('_merge=="left_only"')
print(f"\n90 :\n{rst}")

# merge 컬럼 없애기
rst = pd.merge(df1, df3, how='outer', indicator=True).query('_merge=="left_only"').drop(columns=['_merge'])
print(f"\n91 :\n{rst}")

# (데이터 가공하기)
# 테스트를 위하여 데이터 프레임 생성
df = pd.DataFrame(
    {
        "a": [4, 5, 6, 7],
        "b": [8, 9, 10, 11],
        "c": [12, 13, 14, 15]
    },
    index=[1, 2, 3, 4]
)
print(f"\n92 :\n{df}")

# 행 전체를 한 칸 아래로 이동하기
rst = df.shift(1)
print(f"\n93 :\n{rst}")

# 행 전체를 한 칸 위로 이동하기
rst = df.shift(-1)
print(f"\n94 :\n{rst}")

# 첫 행부터 누적해서 더하기
rst = df.cumsum()
print(f"\n95 :\n{rst}")

# 첫 행부터 누적해서 곱하기
rst = df.cumprod()
print(f"\n96 :\n{rst}")

# 새 행과 이전 행을 비교하면서 최대값을 출력하기
rst = df.cummax()
print(f"\n97 :\n{rst}")

# 새 행과 이전 행을 비교하면서 최소값을 출력하기
rst = df.cummin()
print(f"\n98 :\n{rst}")

# (그룹별 집계하기)
# 실습을 위한 데이터를 CSV 에서 가져옵니다.
df = pd.read_csv("../resources/datasets/23_basic_pandas/house_train.csv")  # 집값 예측용 데이터셋
print(f"\n99 :\n{df}")

# 그룹 지정 및 그룹별 데이터 수 표시
rst = df.groupby(by='YrSold').size()  # 팔린 년도 중심으로 그루핑
print(f"\n100 :\n{rst}")

# 그룹 지정 후 원하는 정보 표시
rst = df.groupby(by='YrSold')['LotArea'].mean()  # 그룹핑 후 각 그룹별 주차장 넓이 표시
print(f"\n101 :\n{rst}")

# 밀집도 기준으로 순위 부여
rst = df['SalePrice'].rank(method='dense')
print(f"\n102 :\n{rst}")

# 최저값 기준으로 순위 부여
rst = df['SalePrice'].rank(method='min')
print(f"\n103 :\n{rst}")

# 순위를 비율로 표시하기
rst = df['SalePrice'].rank(pct=True)
print(f"\n104 :\n{rst}")

# 동일 순위에 대한 처리 방법 정하기
rst = df['SalePrice'].rank(method='first')
print(f"\n105 :\n{rst}")

# object 형 데이터를 숫자 데이터로 수정
# 'WD', 'RD' 와 같은 데이터를 0, 1 과 같이 숫자로 변경하고, One-hot-encoding 을 합니다.
# SaleType 컬럼에 'WD', 'RD' 라는 데이터가 있다면, SaleType 컬럼을 제거하고, WD, RD 라는 컬럼을 생성하고,
# WD 였던 행에는 1, 0, RD 였던 행에는 0, 1 과 같이 표현하여 각 범주에 속할 확률로 변경하는 것입니다.
df = pd.get_dummies(df)
print(f"\n106 :\n{df}")

# 속성별 상관계수 구하기
rst = df.corr()
print(f"\n107 :\n{rst}")
