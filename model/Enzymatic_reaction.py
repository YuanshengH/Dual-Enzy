import torch
import math
import copy
import torch.nn.functional as F
import numpy as np
from torch import nn
from collections.abc import Sequence

def meanpooling(node_emb, mask):
    node_emb = node_emb*(~mask).float().unsqueeze(-1)
    node_emb = node_emb.masked_fill(node_emb==0, np.nan)
    node_emb = node_emb.nanmean(dim=1)
    return node_emb

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Attention(nn.Module):
    def __init__(self, nhead, hidden_size, dropout=0.1, layer_norm_eps=1e-12):
        super().__init__()
        self.num_attention_heads = nhead
        self.hidden_size = hidden_size
        self.attention_head_size = hidden_size // nhead
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attention_dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, self.all_head_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, src_mask=None, tgt_mask=None, output_attentions=False):
        q = self.query(query)       #(bs, l, d)
        k = self.key(key)
        v = self.value(value)

        query_layer = self.transpose_for_scores(q)  #(bs, h, l, d_k)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if (src_mask is not None) and (tgt_mask is None):
            src_mask = src_mask.bool()
            src_mask = src_mask.unsqueeze(1).repeat(1, src_mask.size(-1), 1)
            src_mask = src_mask.unsqueeze(1).repeat(1, attention_scores.size(1), 1, 1)
            attn_mask = src_mask
            attention_scores[attn_mask] = float(-9e9)
        elif (src_mask is not None) and (tgt_mask is not None):
            src_mask = src_mask.bool()
            tgt_mask = tgt_mask.bool()
            attn_mask = tgt_mask.unsqueeze(1).repeat(1, src_mask.size(-1), 1)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, attention_scores.size(1), 1, 1)
            # attention_scores = attention_scores.permute(0, 1, 3, 2)
            attention_scores[attn_mask] = float(-9e9)

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)

        output = torch.matmul(attention_probs, value_layer)
        output = output.transpose(2, 1).flatten(2)
        output = self.dense(output)
        return (output, attention_probs) if output_attentions else (output,)
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, layer_norm_eps=1e-12):
        super(DecoderLayer, self).__init__()
        self.self_attn = Attention(nhead=nhead, hidden_size=d_model)
        self.multihead_attn = Attention(nhead=nhead, hidden_size=d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, src, tgt, src_mask, tgt_mask):
        # self attn
        self_x, self_attention_scores = self._sa_block(src, src_mask)
        self_x = self.norm1(src + self_x)

        cross_x, cross_attention_scores = self._mha_block(self_x, tgt, src_mask, tgt_mask)
        x = self.norm2(self_x + cross_x)
        x = self.norm3(x + self._ff_block(x))
        return x, self_attention_scores, cross_attention_scores
    
    # self-attention block
    def _sa_block(self, x, src_mask):
        output, attention_probs = self.self_attn(x, x, x,
                           src_mask=src_mask,
                           output_attentions=True)
        return self.dropout1(output), attention_probs

    # multihead attention block
    def _mha_block(self, x, mem, src_mask, tgt_mask):
        output, attention_probs = self.multihead_attn(x, mem, mem,
                                src_mask=src_mask,
                                tgt_mask=tgt_mask,
                                output_attentions=True)
        return self.dropout2(output), attention_probs

    # feed forward block
    def _ff_block(self, x):
        x = self.dropout(self.linear2(self.activation(self.linear1(x))))
        return self.dropout3(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, layer_norm_eps=1e-12):
        super(EncoderLayer, self).__init__()
        self.self_attn = Attention(nhead=nhead, hidden_size=d_model, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, src, src_mask):
        # self attn
        x, attention_probs = self._sa_block(src, src_mask)
        x = self.norm1(src + x)
        x = self.norm2(x + self._ff_block(x))
        return x, attention_probs
    
    # self-attention block
    def _sa_block(self, x, src_mask):
        output, attention_probs = self.self_attn(x, x, x,
                           src_mask=src_mask,
                           output_attentions=True)
        return self.dropout1(output), attention_probs

    # feed forward block
    def _ff_block(self, x):
        x = self.dropout(self.linear2(self.activation(self.linear1(x))))
        return self.dropout2(x)

class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, require_attn):
        super(TransformerLayer, self).__init__()
        self.encoder = EncoderLayer(d_model, nhead, dim_feedforward)
        self.decoder = DecoderLayer(d_model, nhead, dim_feedforward)
        self.require_attn = require_attn

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if tgt is not None:
            encoder_out, reactant_self_attn = self.encoder(src, src_mask=src_key_padding_mask)
            decoder_out, enzyme_self_attn, enzyme_cross_attn = self.decoder(tgt, encoder_out, src_mask=tgt_key_padding_mask, tgt_mask=memory_key_padding_mask)
            return (encoder_out, decoder_out, reactant_self_attn, enzyme_self_attn, enzyme_cross_attn) if self.require_attn else (encoder_out, decoder_out,)
        else:
            encoder_out, product_self_attn = self.encoder(src, src_mask=src_key_padding_mask)
            return (encoder_out, product_self_attn) if self.require_attn else (encoder_out)

class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron.
    Note there is no batch normalization, activation or dropout in the last layer.

    Parameters:
        input_dim (int): input dimension
        hidden_dim (list of int): hidden dimensions
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        dropout (float, optional): dropout rate
    """

    def __init__(self, input_dim, hidden_dims, short_cut=False, batch_norm=False, activation="gelu", dropout=None):
        super(MultiLayerPerceptron, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.dims = [input_dim] + hidden_dims
        self.short_cut = short_cut

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 2):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))
        else:
            self.batch_norms = None

    def forward(self, input):
        """"""
        layer_input = input

        for i, layer in enumerate(self.layers):
            hidden = layer(layer_input)
            if i < len(self.layers) - 1:
                if self.batch_norms:
                    x = hidden.flatten(0, -2)
                    hidden = self.batch_norms[i](x).view_as(hidden)
                hidden = self.activation(hidden)
                if self.dropout:
                    hidden = self.dropout(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden

        return hidden   
    
class EnzymaticModel(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=1, dim_feedforward=2048, hidden_dim=1024, out_dim=512, require_attn=False):
        super(EnzymaticModel, self).__init__()
        self.require_attn = require_attn
        mft_layer = TransformerLayer(d_model, nhead, dim_feedforward, require_attn=require_attn)
        self.layers = _get_clones(mft_layer, num_layers)

        self.enzyme_mlp1 = MultiLayerPerceptron(input_dim=1280, hidden_dims=[1024, 512])
        self.molecule_mlp1 = MultiLayerPerceptron(input_dim=512, hidden_dims=[1024, 512])
        self.enzyme_mlp2 = MultiLayerPerceptron(input_dim=512, hidden_dims=[hidden_dim, out_dim])
        self.molecule_mlp2 = MultiLayerPerceptron(input_dim=512, hidden_dims=[hidden_dim, out_dim])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, esm_emb, reactant, product, esm_padding_mask=None, reactant_padding_mask=None,
                 product_padding_mask=None):
        enzyme_emb = self.enzyme_mlp1(esm_emb)
        reactant_emb = self.molecule_mlp1(reactant)
        product_emb = self.molecule_mlp1(product)
        
        if self.require_attn:
            for idx, mod in enumerate(self.layers):
                reactant_emb, enzyme_emb, reactant_self_attn, enzyme_self_attn, enzyme_cross_attn = mod(src=reactant_emb, tgt=enzyme_emb, src_key_padding_mask=reactant_padding_mask,\
                    tgt_key_padding_mask=esm_padding_mask, memory_key_padding_mask=reactant_padding_mask)
                product_emb, product_self_attn = mod(src=product_emb, tgt=None, src_key_padding_mask=product_padding_mask)

            enzyme_emb = self.enzyme_mlp2(enzyme_emb)
            reactant_emb = self.molecule_mlp2(reactant_emb)
            product_emb = self.molecule_mlp2(product_emb)

            reactant_emb = meanpooling(reactant_emb, reactant_padding_mask)
            product_emb = meanpooling(product_emb, product_padding_mask)
            enzyme_emb = meanpooling(enzyme_emb, esm_padding_mask)
            return reactant_emb, enzyme_emb, product_emb, enzyme_self_attn, enzyme_cross_attn, reactant_self_attn, product_self_attn
        else:
            for idx, mod in enumerate(self.layers):
                reactant_emb, enzyme_emb = mod(src=reactant_emb, tgt=enzyme_emb, src_key_padding_mask=reactant_padding_mask,\
                    tgt_key_padding_mask=esm_padding_mask, memory_key_padding_mask=reactant_padding_mask)
                product_emb = mod(src=product_emb, tgt=None, src_key_padding_mask=product_padding_mask)

            enzyme_emb = self.enzyme_mlp2(enzyme_emb)
            reactant_emb = self.molecule_mlp2(reactant_emb)
            product_emb = self.molecule_mlp2(product_emb)

            reactant_emb = meanpooling(reactant_emb, reactant_padding_mask)
            product_emb = meanpooling(product_emb, product_padding_mask)
            enzyme_emb = meanpooling(enzyme_emb, esm_padding_mask)
            return reactant_emb, enzyme_emb, product_emb
        