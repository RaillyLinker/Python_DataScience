import utils.torch_util as tu
import model_layers.recurrent_nn_sentence_classifier.main_model as recurrent_nn_sentence_classifier
import os
import torch
from torch import nn
import pandas as pd
from Korpora import Korpora
from konlpy.tag import Okt
from collections import Counter
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from gensim.models import Word2Vec
from torch import optim

"""
[순환 신경망 모델 문장 분류기]
- RNN, LSTM, GRU 를 이용한 문장 분류기를 만들 것입니다.
    문장을 입력 하면, 해당 문장에 포함된 감정을 분석 하여 분류 하는 모델입니다.
"""


def main():
    # 사용 가능 디바이스
    device = tu.get_gpu_support_device(gpu_support=True)

    # 학습할 네이버 영화 리뷰 데이터셋 말뭉치 로딩
    corpus = Korpora.load("nsmc")

    # 학습 데이터 프레임 추리기
    corpus_train = pd.DataFrame(corpus.train)
    if corpus_train.isnull().values.any():  # NULL 값 존재 유무
        corpus_train = corpus_train.dropna(how='any')  # 결측값이 존재하는 행을 제거

    corpus_test = pd.DataFrame(corpus.test)
    if corpus_test.isnull().values.any():  # NULL 값 존재 유무
        corpus_test = corpus_test.dropna(how='any')  # 결측값이 존재하는 행을 제거

    print(corpus_train.head(5).to_markdown())
    print("Training Data Size :", len(corpus_train))
    print("Testing Data Size :", len(corpus_test))

    # 토크나이징
    tokenizer = Okt()
    train_tokens = [tokenizer.morphs(review) for review in corpus_train.text]
    test_tokens = [tokenizer.morphs(review) for review in corpus_test.text]

    # ex : [['아', '더빙', '..', '진짜', '짜증나네요', '목소리'], ['흠', '...', '오버', '연기', '조차', '가볍지', '않구나'], ...]
    print(train_tokens[:5])

    # 사전 학습된 word2vec
    # word2vec = None
    word2vec = Word2Vec.load("../_by_product_files/gensim_word_2_vec/word2vec.model")

    if word2vec is None:
        # 사전 학습 모델 적용 안함
        def build_vocab(corpus, n_vocab, special_tokens):
            counter = Counter()
            for tokens in corpus:
                counter.update(tokens)
            vocab = special_tokens
            for token, count in counter.most_common(n_vocab):
                vocab.append(token)
            return vocab

        vocab = build_vocab(corpus=train_tokens, n_vocab=5000, special_tokens=["<pad>", "<unk>"])
        token_to_id = {token: idx for idx, token in enumerate(vocab)}

        print(vocab[:10])
        print(len(vocab))

        unk_id = token_to_id["<unk>"]
        train_ids = [
            [token_to_id.get(token, unk_id) for token in review] for review in train_tokens
        ]
        test_ids = [
            [token_to_id.get(token, unk_id) for token in review] for review in test_tokens
        ]

        def pad_sequences(sequences, max_length, pad_value):
            result = list()
            for sequence in sequences:
                sequence = sequence[:max_length]
                pad_length = max_length - len(sequence)
                padded_sequence = sequence + [pad_value] * pad_length
                result.append(padded_sequence)
            return np.asarray(result)

        max_length = 32
        pad_id = token_to_id["<pad>"]
        train_ids = pad_sequences(train_ids, max_length, pad_id)
        test_ids = pad_sequences(test_ids, max_length, pad_id)

        print(train_ids[0])
        print(test_ids[0])

        train_dataset = TensorDataset(
            torch.tensor(train_ids),
            torch.tensor(
                corpus_train.label.values,
                dtype=torch.float32
            ).unsqueeze(1)
        )
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        test_dataset = TensorDataset(
            torch.tensor(test_ids),
            torch.tensor(
                corpus_test.label.values,
                dtype=torch.float32
            ).unsqueeze(1)
        )
        validation_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        embedding_layer = None
        vocab_size = len(token_to_id)
    else:
        # 사전 학습 모델 적용
        embedding_dim = word2vec.vector_size
        id_to_token = word2vec.wv.index_to_key
        vocab_size = len(id_to_token)
        init_embeddings = np.zeros((vocab_size, embedding_dim))

        for index, token in enumerate(id_to_token):
            init_embeddings[index] = word2vec.wv[token]

        embedding_layer = nn.Embedding.from_pretrained(
            torch.tensor(init_embeddings, dtype=torch.float32)
        )

        unk_id = word2vec.wv.key_to_index["<unk>"]
        train_ids = [
            [word2vec.wv.key_to_index.get(token, unk_id) for token in review] for review in train_tokens
        ]
        test_ids = [
            [word2vec.wv.key_to_index.get(token, unk_id) for token in review] for review in test_tokens
        ]

        def pad_sequences(sequences, max_length, pad_value):
            result = list()
            for sequence in sequences:
                sequence = sequence[:max_length]
                pad_length = max_length - len(sequence)
                padded_sequence = sequence + [pad_value] * pad_length
                result.append(padded_sequence)
            return np.asarray(result)

        max_length = 32
        pad_id = word2vec.wv.key_to_index["<pad>"]
        train_ids = pad_sequences(train_ids, max_length, pad_id)
        test_ids = pad_sequences(test_ids, max_length, pad_id)

        print(train_ids[0])
        print(test_ids[0])

        train_dataset = TensorDataset(
            torch.tensor(train_ids),
            torch.tensor(
                corpus_train.label.values,
                dtype=torch.float32
            ).unsqueeze(1)
        )
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        test_dataset = TensorDataset(
            torch.tensor(test_ids),
            torch.tensor(
                corpus_test.label.values,
                dtype=torch.float32
            ).unsqueeze(1)
        )
        validation_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 준비된 데이터와 임베딩 레이어로 모델 생성
    model = recurrent_nn_sentence_classifier.MainModel(
        n_vocab=vocab_size,
        rnn_layer_hidden_dim=64,
        rnn_layer_numbers=2,
        pretrained_embedding_layer=embedding_layer,
        rnn_layer_model_type="gru"  # 사용할 RNN 모델 (rnn, lstm, gru)
    )

    # 모델 학습
    tu.train_model(
        device=device,
        model=model,
        criterion=nn.BCEWithLogitsLoss(),
        optimizer=optim.RMSprop(model.parameters(), lr=0.001),
        train_dataloader=train_dataloader,
        num_epochs=5,
        validation_dataloader=validation_dataloader,
        check_point_file_save_directory_path="../_check_point_files/recurrent_nn_sentence_classifier",
        # check_point_load_file_full_path="../_check_point_files/~/checkpoint(2024_02_29_17_51_09_330).pt",
        log_freq=1
    )

    # 모델 저장
    model_file_save_directory_path = "../_torch_model_files/recurrent_nn_sentence_classifier"
    if not os.path.exists(model_file_save_directory_path):
        os.makedirs(model_file_save_directory_path)
    save_file_full_path = tu.save_model_file(
        model=model,
        model_file_save_directory_path=model_file_save_directory_path
    )

    # # 저장된 모델 불러오기
    model = torch.load(save_file_full_path, map_location=device)
    print("Model Load Complete!")
    print(model)


if __name__ == '__main__':
    main()
