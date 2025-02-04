import torch
from torch import nn
import matplotlib.pyplot as plt
## Encoder
"""
    input shape: [b, seq_len, d_model]
"""
class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        position = torch.arange(0, max_seq_len).unsqueeze(1)

        item = 1/10000 ** (torch.arange(0, d_model, 2) / d_model)

        tmp_pos = position * item

        pe = torch.zeros(max_seq_len, d_model)

        pe[:, 0::2] = torch.sin(tmp_pos)
        pe[:, 1::2] = torch.cos(tmp_pos)

        # plt.matshow(pe)
        # plt.show()

        # 因为有batch，所以要进行升维
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, False)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        pe = self.pe
        return x + pe[:, :seq_len, :]

def attention(query, key, value, mask=None):
    d_model = key.shape[-1]
    # q, k, v 的 shape 都是[b, seq_len, d_model]
    att_ = torch.matmul(query, key.transpos(-2, -1)) / d_model ** 0.5

    if mask is not None:
        att_ = att_.masked_fill_(mask, -1e9)

    att_score = torch.softmax(att_, -1)

    return torch.matmul(att_score, value)

class MultiHeadAttention(nn.Module):
    def __int__(self, heads, d_model, dropout):
        super().__init__()
        assert d_model & heads == 0
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.linear = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.heads = heads
        self.d_k = d_model / heads
        self.d_model = d_model

    def forward(self, q, k, v, mask=None):
        # [n, seq_len, d_model] -> [n, heads, seq_len, d_k]
        q = self.q_linear(q).reshape(q.shape[0], -1, self.heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).reshape(k.shape[0], -1, self.heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).reshape(v.shape[0], -1, self.heads, self.d_k).transpose(1, 2)
        out = attention(q, k, v, mask)
        out = out.transpose(1, 2).reshape(out.shape[0], -1, self.d_model)
        out = self.linear(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x)

class EncoderLayer(nn.Module):
    def __init__(self, heads, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.self_multi_head_att = MultiHeadAttention(heads, d_model, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for i in range(2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        multi_head_att_out = self.self_multi_head_att(x, x, x, mask)
        multi_head_att_out = self.norms[0](x + multi_head_att_out)
        ffn_out = self.ffn(multi_head_att_out)
        ffn_out = self.norms[1](ffn_out + multi_head_att_out)
        out = self.dropout(ffn_out)
        return out

class Encoder(nn.Module):
    def __init__(self, vocab_size, pad_idx, d_model, num_layers, heads, d_ff, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, pad_idx)
        self.pos_encode = PositionEncoding(d_model, max_seq_len)
        self.encode = nn.ModuleList([EncoderLayer(heads, d_model, d_ff, dropout) for i in range(num_layers)])

    def forward(self, x, src_mask=None):
        embed_x = self.embedding(x)
        pos_encode_x = self.pos_encode(embed_x)
        for layer in self.encode:
            pos_encode_x = layer(pos_encode_x, src_mask)
        return pos_encode_x




if __name__ == "__main__":
    # PositionEncoding(512, 100)
    # att = MultiHeadAttention(8, 512, 0.2)
    # x = torch.randn(4, 100, 512)
    # out = att(x, x, x)
    # print(out.shape)