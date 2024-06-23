from Korpora import Korpora
from sentencepiece import SentencePieceTrainer
from sentencepiece import SentencePieceProcessor
import os

"""
[Sentencepiece 토크나이저]
Sentencepiece : 구글에서 개발한 오픈소스 하위 단어 토크나이저 라이브러리
    바이트 페어 인코딩과 유사한 알고리즘을 사용해 입력 데이터를 토큰화하고 단어 사전을 생성합니다.
    또한 워드피스, 유니코드 기반의 다양한 알고리즘을 지원하며, 사용자가 직접 설정할 수 있는 파라미터드을 제공합니다.
말뭉치가 크고 양질일 수록 토크나이저의 성능은 좋아집니다.
"""

# Korpora : 국립 국어원, 혹은 AI Hub 에서 제공하는 한국어 말뭉치 데이터를 쉽게 사용할 수 있게 제공하는 오픈소스 라이브러리
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

# corpus.txt 말뭉치 데이터를 사용하여 토크나이저 모델 학습
print("corpus.txt 말뭉치 데이터를 사용하여 토크나이저 모델 학습")

# SentencePieceTrainer 파라미터 ->
# model_prefix : 모델 학습시 .model 파일과 .vocab 파일이 생성 됩니다. 이에 대한 파일명 입니다.
# vocab_size : 어휘사전 크기
# character_coverage : 말뭉치 내에 존재하는 글자 중 토크나이저가 다룰 수 있는 글자의 비율
# model_type : 토크나이저 알고리즘 (unigram, bpe, char, word)
# max_sentence_length : 최대 문장 길이
# unk_id : 어휘 사전에 없는 OOV 에 할당되는 토큰의 id (기본값 : 0)
# bos_id : 문장이 시작되는 지점을 의미하는 토큰의 id (기본값 : 1)
# eos_id : 문장이 끝나는 지점을 의미하는 토큰의 id (기본값 : 2)
# 이외에는 SentencePiece 깃허브(https://github.com/google/sentencepiece)에서 확인

model_file_save_directory_path = "../_by_product_files/sentencepiece_tokenizer"
if not os.path.exists(model_file_save_directory_path):
    os.makedirs(model_file_save_directory_path)

SentencePieceTrainer.Train(
    "--input=../resources/datasets/global/corpus.txt\
    --model_prefix=../_by_product_files/sentencepiece_tokenizer/petition_bpe\
    --model_type=bpe"
)

# SentencePieceProcessor 에 .model 파일 로딩
tokenizer = SentencePieceProcessor()
tokenizer.load("../_by_product_files/sentencepiece_tokenizer/petition_bpe.model")

# sentencepiece 학습 결과 확인
sentence = "안녕하세요, 토크나이저가 잘 학습되었군요!"
sentences = ["이렇게 입력값을 리스트로 받아서", "쉽게 토크나이저를 사용할 수 있답니다"]
tokenized_sentence = tokenizer.encode_as_pieces(sentence)
tokenized_sentences = tokenizer.encode_as_pieces(sentences)
print("단일 문장 토큰화 :", tokenized_sentence)
print("여러 문장 토큰화 :", tokenized_sentences)
encoded_sentence = tokenizer.encode_as_ids(sentence)
encoded_sentences = tokenizer.encode_as_ids(sentences)
print("단일 문장 정수 인코딩 :", encoded_sentence)
print("여러 문장 정수 인코딩 :", encoded_sentences)
decode_ids = tokenizer.decode_ids(encoded_sentences)
decode_pieces = tokenizer.decode_pieces(encoded_sentences)
print("정수 인코딩에서 문장 변환 :", decode_ids)
print("하위 단어 토큰에서 문장 변환 :", decode_pieces)

# sentencepiece 어휘 사전 확인
vocab = {idx: tokenizer.id_to_piece(idx) for idx in range(tokenizer.get_piece_size())}
print(list(vocab.items())[:5])
print("vocab size :", len(vocab))
