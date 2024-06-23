from torch import nn
import torch


class MainModel(nn.Module):
    def __init__(
            self,
            n_vocab,
            rnn_layer_hidden_dim,
            rnn_layer_numbers,
            rnn_layer_dropout=0.5,
            rnn_layer_bidirectional=True,
            rnn_layer_model_type="gru",
            pretrained_embedding_layer=None
    ):
        super().__init__()
        embedding_dim = 128

        # 모델 내 레이어
        if pretrained_embedding_layer is not None:
            self.embedding = pretrained_embedding_layer
        else:
            self.embedding = nn.Embedding(
                num_embeddings=n_vocab,
                embedding_dim=embedding_dim,
                padding_idx=0
            )

        if rnn_layer_model_type == "rnn":
            self.model = nn.RNN(
                input_size=embedding_dim,
                hidden_size=rnn_layer_hidden_dim,
                num_layers=rnn_layer_numbers,
                bidirectional=rnn_layer_bidirectional,
                dropout=rnn_layer_dropout,
                batch_first=True,
            )
        elif rnn_layer_model_type == "lstm":
            self.model = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=rnn_layer_hidden_dim,
                num_layers=rnn_layer_numbers,
                bidirectional=rnn_layer_bidirectional,
                dropout=rnn_layer_dropout,
                batch_first=True,
            )
        elif rnn_layer_model_type == "gru":
            self.model = nn.GRU(
                input_size=embedding_dim,
                hidden_size=rnn_layer_hidden_dim,
                num_layers=rnn_layer_numbers,
                bidirectional=rnn_layer_bidirectional,
                dropout=rnn_layer_dropout,
                batch_first=True,
            )

        if rnn_layer_bidirectional:
            self.classifier = nn.Linear(rnn_layer_hidden_dim * 2, 1)
        else:
            self.classifier = nn.Linear(rnn_layer_hidden_dim, 1)
        self.dropout = nn.Dropout(rnn_layer_dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)

        for name, param in self.model.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)  # 직교 초기화 적용
            elif 'bias' in name:
                param.data.fill_(0)

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, model_in):
        embeddings = self.embedding(model_in)
        output, _ = self.model(embeddings)
        last_output = output[:, -1, :]
        last_output = self.dropout(last_output)
        model_out = self.classifier(last_output)
        return model_out
