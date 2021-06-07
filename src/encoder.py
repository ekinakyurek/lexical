import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(
            self,
            vocab,
            n_embed,
            n_hidden,
            n_layers,
            bidirectional=True,
            dropout=0,
            rnntype=nn.LSTM,
    ):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(len(vocab), n_embed, vocab.pad())
        self.embed_dropout = nn.Dropout(dropout)
        self.rnn = rnntype(
            n_embed, n_hidden, n_layers, bidirectional=bidirectional, dropout=dropout
        )

    def forward(self, data, lens=None):
        if len(data.shape) == 3:
            emb    = torch.matmul(data, self.embed.weight)
            tokens = torch.argmax(data.detach(),dim=-1)
            emb    = emb * (tokens != self.vocab.pad()).unsqueeze(2).float()
        else:
            emb   = self.embed(data)
        if lens is not None:
            padded_sequence = self.embed_dropout(emb)
            total_length = padded_sequence.shape[0]
            packed_sequence = nn.utils.rnn.pack_padded_sequence(padded_sequence, lens)
            packed_output, hidden = self.rnn(packed_sequence)
            output_padded,_ = nn.utils.rnn.pad_packed_sequence(packed_output,
                                                               total_length=total_length,
                                                               padding_value=self.vocab.pad())
            return output_padded, hidden
        else:
            return self.rnn(self.embed_dropout(emb))
