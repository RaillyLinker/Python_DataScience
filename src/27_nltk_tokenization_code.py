import nltk
from nltk import tokenize
from nltk import tag

"""
[NLTK (Natural Language Toolkit) 란]
- 자연어 처리를 위해 개발된 라이브러리로, 토큰화, 형태소 분석, 구문 분석, 개체명 인식, 감정 분석 등의 기능을 제공 합니다.
    아래 코드는 NLTK 를 사용하여 단어를 토큰화 하는 예시입니다.
"""

sentence = "Those who can imagine anything, can create the impossible."

# punkt, averaged_perceptron_tagger 모델 다운로드
# 두 모델 모두 Treebank 라는 대규모 영어 말뭉치를 기반으로 학습되었습니다.
# punkt 는 통계 기반 모델이며, averaged_perceptron_tagger 는 퍼셉트론 기반 품사 태깅을 수행합니다.
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# 단어 토큰 (공백, 구두점 등의 정보로 단어를 분리합니다.)
word_tokens = tokenize.word_tokenize(sentence)

# 문장 토큰 - 문장 단위로 토큰을 나눕니다. (마침표, 느낌표, 물음표 등의 구두점을 기준으로 문장을 분리합니다.)
sent_tokens = tokenize.sent_tokenize(sentence)

print("word_tokens")
print(word_tokens)
print("")

print("sent_tokens")
print(sent_tokens)
print("")

# 단어 토큰화 후 품사 태깅
word_tokens = tokenize.word_tokenize(sentence)
pos = tag.pos_tag(word_tokens)

print("품사 태깅")
print(pos)
print("")
