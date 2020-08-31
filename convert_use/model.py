import torch
from torch import nn


class Embeddings(nn.Module):
    def __init__(self):
        self.embedding = nn.Embedding(128010, 512)  # ! Recheck.

    def forward(self):
        raise NotImplementedError


class EncoderDNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.DNN = DNN()

    def forward(self):
        raise NotImplementedError


class DNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.ResidualHidden_0 = ResidualHidden(512, 320)
        self.ResidualHidden_1 = ResidualHidden(320, 320, adjust_depth_from=512)
        # self.ResidualHidden_1 = nn.Linear(320, 320)
        # self.AdjustDepth = AdjustDepth(512, 320)

        self.ResidualHidden_2 = ResidualHidden(320, 512)
        self.ResidualHidden_3 = ResidualHidden(512, 512, adjust_depth_from=320)
        # self.hidden_3_proj = AdjustDepth(320, 512)

    def forward(self, x):
        hidden_0 = self.hidden_0(x)
        hidden_1 = self.hidden_1(hidden_0) + self.hidden_1_proj(x)

        hidden_2 = self.hidden_2(hidden_1)
        self.hidden_3(hidden_2) + self.hidden_3_proj(hidden_1)
        raise NotImplementedError


class ResidualHidden(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, adjust_depth_from: int = None):
        super().__init__()

        self.dense = nn.Linear(input_dim, output_dim)
        if adjust_depth_from:
            self.adjust = True
            self.AdjustDepth = AdjustDepth(adjust_depth_from, output_dim)
        else:
            self.adjust = False

    def forward(self):
        raise NotImplementedError


class AdjustDepth(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projection(x)


class EncoderTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.Transformer = Transformer()
        self.hidden_layers = TanhLayerWrapper()

    def forward(self):
        raise NotImplementedError


class Transformer(nn.Module):
    def __init__(self):

        self.SparseTransformerEncode = TransformerEncoder()

        self.layer_prepostprocess = LayerPrePostProcess()

        self.AttentionPooling = AttentionPooling()

    def forward(self):
        raise NotImplementedError


class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.Layer_0 = Encoder(512, 8)
        self.Layer_1 = Encoder(512, 8)
        self.Layer_2 = Encoder(512, 8)
        self.Layer_3 = Encoder(512, 8)
        self.Layer_4 = Encoder(512, 8)
        self.Layer_5 = Encoder(512, 8)

    def forward(self):
        raise NotImplementedError


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.SelfAttention = SelfAttention(d_model, num_heads)
        self.FFN = FFN(d_model)

    def forward(self, x):
        hidden = self.self_attention(x)
        self.feed_forward(hidden)
        raise NotImplementedError


class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.SparseMultiheadAttention = SparseMultiheadAttention(d_model, num_heads)
        self.layer_prepostprocess = LayerPrePostProcess(d_model)

    def forward(self, x):
        context = self.attention(x)
        self.output_norm(context)
        raise NotImplementedError


class LayerPrePostProcess(nn.Module):
    def __init__(self, d_model):
        super(LayerPrePostProcess, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        self.layer_norm(x)
        raise NotImplementedError


class SparseMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(SparseMultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        self.ComputeQKV = ComputeQKV(d_model, num_heads)

    def forward(self, x):
        self.ComputeQKV(x, x, x)
        raise NotImplementedError


class ComputeQKV(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.d_model = d_model

        self.depth = d_model // self.num_heads

        self.compute_q = nn.Linear(d_model, d_model)
        self.compute_k = nn.Linear(d_model, d_model)
        self.compute_v = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        split_x_shape = x.size()[:-1] + (self.num_heads, self.depth)
        x = x.view(*split_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value):
        query = self.compute_q(query)
        key = self.compute_k(key)
        value = self.compute_v(value)

        query_layer = self.transpose_for_scores(query)
        key_layer = self.transpose_for_scores(key)
        value_layer = self.transpose_for_scores(value)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores /= torch.sqrt(self.depth)
        attention_probs = self.softmax(attention_scores)

        intermediate_context_layer = torch.matmul(attention_probs, value_layer)
        intermediate_context_layer = intermediate_context_layer.permute(
            0, 2, 1, 3
        ).contiguous()
        context_layer_shape = intermediate_context_layer.size()[:-2] + (self.d_model,)
        context_layer = intermediate_context_layer.view(*context_layer_shape)
        return context_layer


class FFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.Layer1 = nn.Linear(d_model, d_model * 4)
        self.Layer2 = nn.Linear(d_model * 4, d_model)
        self.layer_prepostprocess = LayerPrePostProcess(d_model)

    def forward(self, x):
        hidden = self.Layer1(x)
        hidden = self.Layer2(hidden)
        # return self.layer_norm(x, hidden)
        raise NotImplementedError


class AttentionPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.AttentionPooling = None
        self.AttentionLogits = None
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError


class TanhLayerWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh_layer_0 = TanhLayer()

    def forward(self):
        raise NotImplementedError


class TanhLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Tanh()

    def forward(self):
        raise NotImplementedError
