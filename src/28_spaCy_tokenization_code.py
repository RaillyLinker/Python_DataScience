import spacy

"""
[spaCy 란]
- Cython 기반으로 개발된 오픈 소스 라이브러리로서, NLTK 라이브러리와 마찬가지로 자연어 처리 관련 기능을 제공합니다.
    장점은, NLTK 보다 빠른 속도와 높은 정확도를 목표로 한 머신러닝 기반의 자연어 처리 기능을 제공합니다.
    단점은, NLTK 모델보다 크고 더 많은 리소스를 요구합니다.
"""

sentence = "Those who can imagine anything, can create the impossible."

# NLP 로딩
# 영어로 사전 학습된 모델인 en_core_web_sm 를 다운로드 하기 위해선
# >> python -m spacy download en_core_web_sm
# 입력 후,
nlp = spacy.load("en_core_web_sm")

# NLP 모델로 문장을 토크나이징
doc = nlp(sentence)

# 품사 태깅과 같이 토크나이징 결과 로깅
for token in doc:
    print(f"[{token.pos_:5} - {token.tag_:3}] : {token.text}")
