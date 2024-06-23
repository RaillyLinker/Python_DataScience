from Korpora import Korpora
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import Sequence, NFD, Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import WordPiece as WordPieceDecoder
import os

"""
[tokenizers]
아래 코드는 tokenizers 라이브러리를 사용하여 word piece 토크나이징을 실행하는 방법을 정리한 것입니다.
"""

# Korpora : 국립 국어원, 혹은 AI Hub 에서 제공하는 한국어 말뭉치 데이터를 쉽게 사용할 수 있게 제공하는 오픈소스 라이브러리
# 말뭉치가 크고 양질일 수록 토크나이저의 성능은 좋아집니다.

print("Korpora 말뭉치 목록 확인")
print(Korpora.corpus_list())
print("")

# 청와대 청원 데이터 korean_petitions 로딩
print("청와대 청원 데이터 korean_petitions 로딩")
corpus = Korpora.load("korean_petitions")

# 학습 데이터 가져오기
dataset = corpus.train

print(f"\n청와대 청원 데이터 리스트 개수 {len(dataset)}\n")

# 데이터 1행을 가져오기
petition = dataset[0]

print("청원 시작일 :", petition.begin)
print("청원 종료일 :", petition.end)
print("청원 동의 수 :", petition.num_agree)
print("청원 범주 :", petition.category)
print("청원 제목 :", petition.title)
print("청원 본문 :", petition.text[:30])

# 청원 데이터를 텍스트 데이터로 저장
print("청원 데이터를 텍스트 데이터로 저장")
petitions = corpus.get_all_texts()
with open("../resources/datasets/global/corpus.txt", "w", encoding="utf-8") as f:
    for petition in petitions:
        # 파일 내에 한줄씩 데이터 저장
        f.write(petition + "\n")

model_file_save_directory_path = "../_by_product_files/tokenizers_word_piece_tokenizer"
if not os.path.exists(model_file_save_directory_path):
    os.makedirs(model_file_save_directory_path)

# corpus.txt 말뭉치 데이터를 사용하여 토크나이저 모델 학습
print("corpus.txt 말뭉치 데이터를 사용하여 토크나이저 모델 학습")
tokenizer = Tokenizer(WordPiece())
tokenizer.normalizer = Sequence([NFD(), Lowercase()])
tokenizer.pre_tokenizer = Whitespace()
tokenizer.train(["../resources/datasets/global/corpus.txt"])
tokenizer.save("../_by_product_files/tokenizers_word_piece_tokenizer/petition_wordpiece.json")

# 토크나이저 모델을 불러오고 테스트
tokenizer = Tokenizer.from_file("../_by_product_files/tokenizers_word_piece_tokenizer/petition_wordpiece.json")
tokenizer.decoder = WordPieceDecoder()
sentence = "안녕하세요, 토크나이저가 잘 학습되었군요!"
sentences = ["이렇게 입력값을 리스트로 받아서", "쉽게 토크나이저를 사용할 수 있답니다"]
encoded_sentence = tokenizer.encode(sentence)
encoded_sentences = tokenizer.encode_batch(sentences)
print("인코더 형식 :", type(encoded_sentence))
print("단일 문장 토큰화 :", encoded_sentence.tokens)
print("여러 문장 토큰화 :", [enc.tokens for enc in encoded_sentences])
print("단일 문장 정수 인코딩 :", encoded_sentence.ids)
print("여러 문장 정수 인코딩 :", [enc.ids for enc in encoded_sentences])
print("정수 인코딩에서 문장 변환 :", tokenizer.decode(encoded_sentence.ids))
