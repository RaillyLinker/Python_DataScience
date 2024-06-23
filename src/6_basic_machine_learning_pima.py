import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, BaggingClassifier, \
    StackingClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def main():
    """
    [머신러닝 알고리즘]
    - 최근에는 딥러닝이 머신러닝 알고리즘 중 으뜸으로 여겨지는데,
        딥러닝은 비선형 데이터 분석 및 인공지능을 구현하는데에 있어 가장 효과적인 방법인 것이지,
        데이터분석 전 분야에 있어서 딥러닝 그 자체는 만능이 아닙니다.
        예를들어 정말 단순한 선형 회귀 모델로 설명이 가능한 현상이 있을 때,
        이를 딥러닝으로 푼다면 그 구조상 쓸모없는 연산이 많을 수 밖에 없습니다.
        1번 계산해도 될 것을 1000번 넘게 계산해야 한다면 최적화가 필요한 위치에는 사용할 수 없을 것입니다.
        머신러닝에는 딥러닝 이외에도 많은 알고리즘이 존재하며, 상황에 따라서 적절히 사용하면 최적의 효과를 낼 수 있습니다.
        아마 딥러닝이 인공지능을 완성한 이후에도, 머신러닝은 사장되지 않으며,
        오히려 딥러닝 인공지능에 의해 사용되는 도구로 더욱 발전할 것이라고 생각이 됩니다.
        인간으로써 이를 이해하고 사용하기 위하여 정리해봅니다.

    - 사이킷런(Scikitlearn)
        파이썬의 대표적인 머신러닝 라이브러리입니다.
        오픈소스 라이브러리로, 여러 머신러닝 알고리즘이 구현되어 함수로 제공되며,
        사이킷런의 머신러닝 알고리즘을 사용하는 기본 순서는,
        1. 사용할 적절한 알고리즘을 불러오기.
        2. fit() 함수로 데이터를 학습
        3. predict() 함수로 예측
        공통적으로 위와 같은 절차로 수행하면 됩니다.

    - 실제 문제 해결을 진행하며 알아보겠습니다.
        아래서 풀어볼 문제는, 피마 인디언 데이터(Pima Indians Data)를 분석하는 것으로,
        1950 년대까지만 하더라도 비만 인구가 단 한명도 존재하지 않던 피마 인디언 부족이,
        1980 년대에는 전체 부족의 60% 가 당뇨, 80% 가 비만이라고 합니다.
        이는 영양 섭취가 어려웠던 시절에, 생존을 위하여 영양분을 체내에 저장하는 능력이 강해지도록 진화된 부족이,
        미국의 기름진 패스트 푸드 문화를 만나며 한 세대도 안되어 단기간에 일어난 현상입니다.
        당시에는 원인을 알 수 없었기에 부족 전체의 당뇨 발병 원인을 알기 위하여, 총 768 명의 부족민을 관찰하여,
        개인별 관측 데이터와 당뇨 여부를 기록하여 원인을 알아보려 한 것이 바로 피마 인디언 데이터셋입니다.
        딥러닝을 사용하지 않고, 다양한 머신러닝 알고리즘으로 당뇨 발명 여부를 분류하는 예측모델을 만들어보겠습니다.
    """

    # (데이터 준비 및 확인)
    # CSV 에서 피마 데이터셋 가져오기
    # 피마 데이터셋의 컬럼은,
    # 1. pregnant : 과거 임신 횟수
    # 2. plasma : 포도당 부하 검사 2시간 후 공복 혈당 농도
    # 3. pressure : 확장기 혈압(mm Hg)
    # 4. thickness : 삼두근 피부 주름 두께(mm)
    # 5. insulin : 혈청 인슐린(2-hour, mu U/ml)
    # 6. bmi : 체질량 지수(BMI, weight in kg/(height in m)^2)
    # 7. pedigree : 당뇨병 가족력
    # 8. age : 나이
    # 9. diabetes : 당뇨병 여부(1 : true, 0 : false)
    # 위와 같은 의미를 지닙니다.
    df = pd.read_csv('../resources/datasets/6_basic_machine_learning_pima/pima-indians-diabetes3.csv')

    print("(데이터 프레임 첫 다섯줄 출력)")
    print(df.head(5))

    print("\n(종속변수 그룹별 데이터 개수)")
    print(df["diabetes"].value_counts())
    # 정상인 500 명, 당뇨병 268 명으로, 총 768 개의 샘플 확인

    print("\n(특징별 기본 통계값 확인)")
    print(df.describe())

    print("\n(특징간 상관계수 출력)")
    corr = df.corr()
    print(corr)

    # 상관관계 시각화
    plt.figure(figsize=(12, 12))
    sns.heatmap(corr, linewidths=0.1, vmax=0.5, linecolor='white', annot=True)
    plt.show()

    # (데이터 전처리)
    # 이제 실질적으로 분석 모델을 만들기 전에 데이터를 전처리 하겠습니다.

    # 특징 컬럼 8개는 독립변수 X 로 할당하고, 마지막 당뇨 여부는 종속변수 y 로 할당합니다.
    X = df.iloc[:, 0:8]
    y = df.iloc[:, 8]

    # 각 특징 값의 수치는 스케일이 서로 다르므로, 이를 일정한 범위 안으로 모아놓는 데이터 스케일링을 수행합니다.
    ss = StandardScaler()
    scaled_X = pd.DataFrame(ss.fit_transform(X), columns=X.columns)

    # X 와 scaled_X 를 시각적으로 확인 해 보겠습니다.
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    X.plot(kind='kde', title='Raw data', ax=ax[0])
    scaled_X.plot(kind='kde', title='StandardScaler', ax=ax[1])
    plt.show()
    # Raw X 는 데이터가 너무 편협하게 분포되어 있고, 스케일 차이도 큽니다.
    # 스케일링을 거친 scaled_X 는 데이터가 0 을 중심으로 모여있으며, 데이터 스케일 차이도 크지 않은 것을 확인 할 수 있습니다.

    # 데이터를 학습용과 테스트용으로 75:25 배율로 분배합니다.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    """
    (결정트리(Decision Tree))
    - 결정트리는, 예/아니오 라는 답이 나오는 질문을 계속해서 묻는 구조를 가집니다.
        다중 클래스 분류 문제는 이진 분류의 반복으로 해결할 수 있다는 것과 같으며,
        예를들어 개, 닭, 고등어, 돌고래라는 네 클래스를 분류하는 문제가 있다면,
        육지에 사는지 아닌지에 대한 특징으로 분류,
        육지에 산다면 깃털이 있는지 아닌지로 분류(있다면 닭, 없다면 개)
        육지에 살지 않는다면 아가미 호흡을 하는지 아닌지로 분류(아가미 호흡을 한다면 고등어, 아니라면 돌고래)
        위와 같습니다.
    """
    # 사이킷런 결정트리 객체 생성
    classifier = DecisionTreeClassifier()

    # 모델 학습
    classifier.fit(X_train, y_train)

    # 모델 검증
    accuracies = cross_val_score(estimator=classifier, X=X_test, y=y_test,
                                 cv=StratifiedKFold(n_splits=10, shuffle=True))

    print("\n(DecisionTree Test)")
    print("Accuracy: {:.2f}".format(accuracies.mean() * 100))
    print("Standard Deviation: {:.2f}".format(accuracies.std() * 100))

    """
    (랜덤 포레스트(Random Forest))
    - 랜덤 포레스트는 앞서 정리한 결정 트리를 여러개 묶어둔 것입니다. (나무(Tree)가 모이면 숲(Forest)이되죠.)
        하나의 결정 트리가 아닌, 많은 수의 결정 트리를 실행해, 그로부터 나온 분류 결과를 취합하여, 최종 예측을 하는 알고리즘입니다.
        복잡한 현상에 대해 너무 명확하고 단순한 방법으로 결정을 내리기에 과적합 되기 쉬운 결정 트리의 단점을 보완해 줍니다.
        랜덤 포레스트는 내부적으로 결정 트리라는 다른 머신러닝 모델의 결합으로 이루어지므로,
        앙상블(Ensemble) 모델이라 할 수 있습니다.
    
    - 앙상블이란, 여러 머신러닝 모델을 이용하여 하나의 강력한 모델을 만드는 기법으로, 랜덤포레스트가 그러하며,
        이외에도 Voting, Bagging, Boosting, Stacking 등의 다양한 종류의 앙상블 모델이 존재합니다.
        
    - 랜덤 포레스트의 구조는,
        동일한 구조의 결정 트리가 n 개 준비된 상황에서 동일한 데이터를 입력하는데,
        각 결정 트리의 결과값을 모아서 이를 평균을 낸 값으로 최종 예측을 수행합니다.
    """
    # 결정트리 50개를 모은 구조의 랜덤 포레스트 객체 생성
    classifier = RandomForestClassifier(n_estimators=50)
    # 학습
    classifier.fit(X_train, y_train)

    # 모델 검증
    accuracies = cross_val_score(estimator=classifier, X=X_test, y=y_test,
                                 cv=StratifiedKFold(n_splits=10, shuffle=True))

    print("\n(RandomForest Test)")
    print("Accuracy: {:.2f}".format(accuracies.mean() * 100))
    print("Standard Deviation: {:.2f}".format(accuracies.std() * 100))

    """
    (가우시안 나이브 베이즈(Gaussian Naive Bayes))
    - 가우시안 나이브 베이즈는, 속성이 연속형 변수일 때, 베이즈 정리를 기반으로 각 범주에 속할 확률을 계산하는 방법입니다.
    
    - 예를들어 키를 측정한 데이터가 있다고 할 때, 해당 데이터를 보고 남성인지 여성인지를 예측하는 분류 문제가 있다고 합시다.
        남성의 키와 여성의 키는 각자 평균과 표준편차를 가질 것입니다.
        평균을 중심으로, 표준편차로 데이터가 퍼져 있을 것이며, 남성, 여성인지에 따라 이 확률 분포가 다를 것입니다.
        이때, 목표 데이터를 각 확률분포에 대입하였을 때, 각 범주에 속할 확률이 나올 것이고,
        이 중 가장 높은 확률을 가지는 범주로 데이터를 분류하는 것입니다. 
        (= A 클래스의 확률 변수에서 해당 값이 나올 확률과 B 클래스의 확률 변수에서 해당 값이 나올 확률을 비교)
    """
    # 가우시안 나이브 베이즈 객체 생성
    classifier = GaussianNB()
    # 학습
    classifier.fit(X_train, y_train)

    # 모델 검증
    accuracies = cross_val_score(estimator=classifier, X=X_test, y=y_test,
                                 cv=StratifiedKFold(n_splits=10, shuffle=True))

    print("\n(Gaussian Naive Bayes Test)")
    print("Accuracy: {:.2f}".format(accuracies.mean() * 100))
    print("Standard Deviation: {:.2f}".format(accuracies.std() * 100))

    """
    (K-최근접 이웃(K-Nearest Neighbor))
    - k-최근접 이웃은, 새로운 데이터가 입력되면 가장 가까이 있는 것을 중심으로 새로운 데이터의 종류를 정해주는 것입니다.
        특징이 결과를 잘 설명해주는 데이터라고 한다면,
        동일한 클래스의 경우는 비슷한 특징을 지니므로 데이터들 끼리 서로 가까울 것입니다.
        이러한 데이터 공간 속에, 아직 클래스가 결정되지 않은 신규 데이터가 등장하여 해당 데이터를 분류하기 위해서는, 
        해당 데이터의 주변에 어떤 클래스의 데이터가 더 많은지를 확인하면 됩니다.
        여기서, '주변'이라는 것을 구체화한다면, 분류하고자 하는 데이터의 가장 가까운 데이터를 몇개 추려오는 것입니다. 
        이를 K 라고 합니다.
        가장 가까운 K 개의 데이터를 선별하여, 어떤 클래스에 속하는 데이터가 가장 많은지를 확인하면 됩니다.
    """
    # K-최근접 이웃 객체 생성(k 를 5 로 놓고 분류)
    classifier = KNeighborsClassifier(n_neighbors=5)
    # 학습
    classifier.fit(X_train, y_train)

    # 모델 검증
    accuracies = cross_val_score(estimator=classifier, X=X_test, y=y_test,
                                 cv=StratifiedKFold(n_splits=10, shuffle=True))

    print("\n(K-Nearest Neighbor Test)")
    print("Accuracy: {:.2f}".format(accuracies.mean() * 100))
    print("Standard Deviation: {:.2f}".format(accuracies.std() * 100))

    """
    (에이다 부스트(Ada Boost))
    - 에이다 부스트는, 여러번의 분류를 통해 정답을 예측해가는 알고리즘입니다.
        앞서 랜덤 포레스트의 설명에서 언급했던 부스팅 앙상블 분류기에 속하는 알고리즘으로,
        이전 분류기와 다음 분류기의 결과를 서로 연결하여 하나의 모델로 만드는 것이 Boosting 방식입니다.
    """
    # 에이다 부스트 객체 생성
    classifier = AdaBoostClassifier(algorithm='SAMME')
    # 학습
    classifier.fit(X_train, y_train)

    # 모델 검증
    accuracies = cross_val_score(estimator=classifier, X=X_test, y=y_test,
                                 cv=StratifiedKFold(n_splits=10, shuffle=True))

    print("\n(Ada Boost Test)")
    print("Accuracy: {:.2f}".format(accuracies.mean() * 100))
    print("Standard Deviation: {:.2f}".format(accuracies.std() * 100))

    """
    (XG 부스트(xg Boost))
    """
    # XG 부스트 객체 생성
    classifier = xgb.XGBClassifier()
    # 학습
    classifier.fit(X_train, y_train)

    # 모델 검증
    accuracies = cross_val_score(estimator=classifier, X=X_test, y=y_test,
                                 cv=StratifiedKFold(n_splits=10, shuffle=True))

    print("\n(xg Boost Test)")
    print("Accuracy: {:.2f}".format(accuracies.mean() * 100))
    print("Standard Deviation: {:.2f}".format(accuracies.std() * 100))

    """
    (이차 판별 분석(Quadratic Discriminant Analysis))
    - 이차 판별 분석은, 각 클래스 간의 경계를 결정해 분류하는 방법입니다.
        각 클래스의 분포가 있을 때, 클래스간의 경계가 직선이라면 선형 판별 분석이고,
        2차 방정식에 의한 곡선이라면 이차 판별 분석이 됩니다.
    """
    # 이차 판별 분석 객체 생성
    classifier = QuadraticDiscriminantAnalysis()
    # 학습
    classifier.fit(X_train, y_train)

    # 모델 검증
    accuracies = cross_val_score(estimator=classifier, X=X_test, y=y_test,
                                 cv=StratifiedKFold(n_splits=10, shuffle=True))

    print("\n(Quadratic Discriminant Analysis Test)")
    print("Accuracy: {:.2f}".format(accuracies.mean() * 100))
    print("Standard Deviation: {:.2f}".format(accuracies.std() * 100))

    """
    (서포트 벡터 머신(Support Vector Machine : SVM))
    - 서포트 벡터 머신이란, 분류를 위한 기준선을 정의하는 모델입니다.
    """
    # Support Vector Classifier 객체 생성
    classifier = SVC(kernel="linear")
    # 학습
    classifier.fit(X_train, y_train)

    # 모델 검증
    accuracies = cross_val_score(estimator=classifier, X=X_test, y=y_test,
                                 cv=StratifiedKFold(n_splits=10, shuffle=True))

    print("\n(Support Vector Machine Test)")
    print("Accuracy: {:.2f}".format(accuracies.mean() * 100))
    print("Standard Deviation: {:.2f}".format(accuracies.std() * 100))

    """
    (서포트 벡터 머신 - RBF 커널(Support Vector Machine Using Radial Basis Functions Kernel))
    - 앞서 정리한 서포트 벡터 머신은 직선으로 클래스를 구분하기에, 선형적으로 분리할 수 없는 형태를 지닌 데이터에는 적용할 수 없습니다.
        RBF 커널 방식은 이러한 문제를 해결하기 위해 만들어졌습니다.
    - RBF 커널의 기본 컨셉은, 차원을 늘리는 것입니다.
        2차원에서는 선형적으로 분리할 수 없는 문제라고 하더라도, 한차원을 더 높인다면, 선형으로도 충분히 데이터를 분리할 수 있게 됩니다.
        이때, 차원을 늘릴 때에는 클래스를 기반으로, 클래스의 중심을 기준으로 완만하게 증가하는 법칙을 만들어,
        새로운 데이터가 들어왔을 때에는 기존의 특징을 비교하여 새로운 차원에서의 값을 계산해내고, 이를 선형 모델로 분류해내면 됩니다.
    """
    # Support Vector Classifier RBF 객체 생성
    classifier = SVC(kernel="rbf")
    # 학습
    classifier.fit(X_train, y_train)

    # 모델 검증
    accuracies = cross_val_score(estimator=classifier, X=X_test, y=y_test,
                                 cv=StratifiedKFold(n_splits=10, shuffle=True))

    print("\n(Support Vector Machine RBF Test)")
    print("Accuracy: {:.2f}".format(accuracies.mean() * 100))
    print("Standard Deviation: {:.2f}".format(accuracies.std() * 100))

    """
    (배깅(Bagging))
    - 배깅은 데이터 세트를 여러개로 분리해 분류를 실행하는 방법입니다.
        보팅과의 차이점은, 하나의 알고리즘을 사용한다는 것이고,
        부스팅과의 차이점은, 각 분류기를 제각각 따로따로 분류한다는 것입니다.
        단일 분류기를 여러번 사용함으로써 정확도를 높이고, 과적합을 방지하는 효과가 있습니다.
        
    - 분류기를 n번 반복해서 학습한다면 그때마다 학습셋, 데이터셋을 새롭게 만듭니다.
        맨 처음 데이터가 n번의 서로 다른 학습셋, 데이터셋으로 분리될 때는, 
        학습셋과 테스트셋을 설정 기준에 따라 랜덤으로 선택해 만드는 Bootstrap 기법을 사용합니다.
    """
    # Bagging 객체 생성(동일한 SVC 알고리즘을 10 개 사용합니다.)
    classifier = BaggingClassifier(estimator=SVC(kernel="rbf"), n_estimators=10)
    # 학습
    classifier.fit(X_train, y_train)

    # 모델 검증
    accuracies = cross_val_score(estimator=classifier, X=X_test, y=y_test,
                                 cv=StratifiedKFold(n_splits=10, shuffle=True))

    print("\n(Bagging Test)")
    print("Accuracy: {:.2f}".format(accuracies.mean() * 100))
    print("Standard Deviation: {:.2f}".format(accuracies.std() * 100))

    """
    (보팅(Voting))
    - 보팅은, 여러가지 다른 유형의 알고리즘을 같은 데이터셋에 적용해 학습하는 방법입니다.
        학습 결과를 모아 다수의 분류기가 결정한 결과를 선택하거나 클래스별 평균을 종합해 예측합니다.
        
    - 각 알고리즘별 최대한 학습시킨 에이다 부스트, 랜덤 포레스트, 서포트 벡터 머신, 이차 판별 분석 모델이 존재한다고 합시다.
    
        에이다 부스트의 결과, 클래스 1 : 68%, 클래스 2 : 21%, 클래스 3 : 11%
        랜덤 포레스트의 결과, 클래스 1 : 3%, 클래스 2 : 90%, 클래스 3 : 7%
        서포트 벡터 머신의 결과, 클래스 1 : 21%, 클래스 2 : 75%, 클래스 3 : 4%
        이차 판별 분석의 결과, 클래스 1 : 5%, 클래스 2 : 89%, 클래스 3 : 6%
        
        위와 같은 결과가 나왔다고 가정하면,
        이 결과값으로 평균을 내면,
        클래스 1 : 24.3%, 클래스 2 : 68.8%, 클래스 3 : 7%
        이므로, 결과적으로 클래스 2 에 속한다고 결론을 내릴 수 있습니다.
        
    - 위 예시에서 보았듯, 다양한 알고리즘이 있고, 그 결과값을 마치 투표하듯 반환하면,
        이를 종합하여 결과를 도출해내므로, 단일 알고리즘의 단점을 완화시킬 수 있을 것입니다.
    """
    # Voting 에 사용할 모델들을 준비
    clf1 = AdaBoostClassifier(algorithm='SAMME')
    clf2 = RandomForestClassifier()
    clf3 = SVC(kernel="linear")

    # Voting 객체 생성
    classifier = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)])
    # 학습
    classifier.fit(X_train, y_train)

    # 모델 검증
    accuracies = cross_val_score(estimator=classifier, X=X_test, y=y_test,
                                 cv=StratifiedKFold(n_splits=10, shuffle=True))

    print("\n(Voting Test)")
    print("Accuracy: {:.2f}".format(accuracies.mean() * 100))
    print("Standard Deviation: {:.2f}".format(accuracies.std() * 100))

    """
    (스택킹(Stacking))
    - 스택킹은, 보팅과 같이 서로 다른 모델에서 나오는 결과값을 사용하는 알고리즘인데,
        결과값을 단순히 평균내는 것이 아니라, 결과값을 쌓아서(Stacking) 새로운 결과 벡터를 만들어내고,
        이 결과 벡터를 입력값으로 하여 최종 결과를 분류해내는 최종 모델을 실행시키는 방식으로 앙상블을 구현합니다.
    """
    # Stacking 에 사용할 모델들을 준비
    clf1 = AdaBoostClassifier(algorithm='SAMME')
    clf2 = RandomForestClassifier()
    clf3 = SVC(kernel="linear")
    clf_final = SVC(kernel="rbf")

    # Stacking 객체 생성
    classifier = StackingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], final_estimator=clf_final)
    # 학습
    classifier.fit(X_train, y_train)

    # 모델 검증
    accuracies = cross_val_score(estimator=classifier, X=X_test, y=y_test,
                                 cv=StratifiedKFold(n_splits=10, shuffle=True))

    print("\n(Stacking Test)")
    print("Accuracy: {:.2f}".format(accuracies.mean() * 100))
    print("Standard Deviation: {:.2f}".format(accuracies.std() * 100))

    """
    - 위와 같이 피마 인디언 데이터셋을 이용하여 분류 문제를 해결하는 머신러닝 라이브러리들을 알아보았습니다.
        코드를 확인하면, 머신러닝 객체 생성 -> fit -> cross_val_score 의 함수가 반복되어,
        머신러닝 알고리즘을 바꾸려면 객체 생성 부분만을 변경하면 되는 것을 알 수 있는데,
        이렇게 동일한 구조를 재활용 가능한 것이 사이킷런의 이점이며, 이를 API 설계가 잘 되었다고 표현을 합니다.
        
    - 위와 같이 많은 머신러닝 모델이 있는데, 어떤 것이 좋은 모델일까요?
        모델간 절대적인 우열은 없지만, 상황에 따라, 데이터에 따라 유리한 모델을 고르는 것이 과제일 것입니다.
        마지막으로, 앞서 사용한 여러 알고리즘의 성능을 한눈에 비교하기 위한 데이터 시각화를 해보겠습니다.
    """

    print("\n 머신러닝 모델 종합 평가")
    # 평가할 모델 객체 리스트
    classifiers = {
        'D_Tree': DecisionTreeClassifier(),
        'RF': RandomForestClassifier(n_estimators=50),
        'GNB': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Ada': AdaBoostClassifier(algorithm='SAMME'),
        'XG': xgb.XGBClassifier(),
        'QDA': QuadraticDiscriminantAnalysis(),
        'SVM_L': SVC(kernel="linear"),
        'SVM_R': SVC(kernel="rbf"),
        'Bagging': BaggingClassifier(estimator=SVC(kernel="rbf"), n_estimators=10),
        'Voting':
            VotingClassifier(
                estimators=[('lr', AdaBoostClassifier(algorithm='SAMME')), ('rf', RandomForestClassifier()),
                            ('gnb', SVC(kernel="linear"))]),
        'Stacking':
            StackingClassifier(
                estimators=[('lr', AdaBoostClassifier(algorithm='SAMME')), ('rf', RandomForestClassifier()),
                            ('gnb', SVC(kernel="linear"))], final_estimator=SVC(kernel="rbf"))
    }

    model_names = []
    model_accuracies = []
    model_means = []

    for classifier_key in classifiers.keys():
        classifier = classifiers[classifier_key]
        accuracies = cross_val_score(estimator=classifier, X=X_test, y=y_test,
                                     cv=StratifiedKFold(n_splits=10, shuffle=True))
        print("Mean accuracy of", classifier_key, ": {: .2f}".format(accuracies.mean() * 100))
        model_names.append(classifier_key)
        model_accuracies.append(accuracies)
        model_means.append(accuracies.mean() * 100)

    # 모델 결과 시각화
    # 막대 그래프
    plt.figure(figsize=(10, 5))
    plt.ylim([60, 80])
    plt.bar(model_names, model_means)

    # 박스 그래프
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.boxplot(model_accuracies)
    ax.set_xticklabels(model_names)

    plt.show()


if __name__ == '__main__':
    main()
