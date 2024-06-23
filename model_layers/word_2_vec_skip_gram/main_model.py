from torch import nn


class MainModel(nn.Module):
    def __init__(
            self,
            # 단어를 나타내는 one-hot-vector 를 만들기 위한 단어 사전 크기
            vocab_size,
            # 단어 의미를 뜻하는 임베딩 벡터의 크기
            embedding_dim
    ):
        super().__init__()
        # 모델 내 레이어
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        self.linear = nn.Linear(
            in_features=embedding_dim,
            out_features=vocab_size
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0.01)

    def forward(self, model_in):
        model_out = self.embedding(model_in)
        model_out = self.linear(model_out)  # vocab_size 의 one-hot-vector 반환
        return model_out
