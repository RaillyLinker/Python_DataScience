from torch import nn
import torch


class MainModel(nn.Module):
    def __init__(
            self,
            n_vocab,
            max_length,
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

        # 모델 내 레이어
        filter_sizes = [3, 3, 4, 4, 5, 5]
        conv = []
        for size in filter_sizes:
            conv.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=embedding_dim,
                        out_channels=1,
                        kernel_size=size
                    ),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=max_length - size - 1),
                )
            )
        self.conv_filters = nn.ModuleList(conv)

        output_size = len(filter_sizes)
        self.pre_classifier = nn.Linear(output_size, output_size)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(output_size, 1)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.layer[0].weight)
        self.layer[0].bias.data.fill_(0.01)

    def forward(self, model_in):
        embeddings = self.embedding(model_in)
        embeddings = embeddings.permute(0, 2, 1)

        conv_outputs = [conv(embeddings) for conv in self.conv_filters]
        concat_outputs = torch.cat([conv.squeeze(-1) for conv in conv_outputs], dim=1)

        logits = self.pre_classifier(concat_outputs)
        logits = self.dropout(logits)
        logits = self.classifier(logits)
        return logits
