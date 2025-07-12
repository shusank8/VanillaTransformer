import torch 
import torch.nn as nn

# model implementation

class InputEmbeddings(nn.Module):
    
    """
    Takes (B, S) and adds C dim 
    (B, S) => (B, S, C)
    """
    def __init__(self, vocab_size, embdim):
        super().__init__()
        self.embdim = embdim
        self.embeddings = nn.Embedding(vocab_size, embdim)
    
    def forward(self, x):
        # diving by (self.embdim)**(1/2) for nice numbers/to keep var small
        return self.embeddings(x) / (self.embdim)**(1/2)


class PositionalEmbeddings(nn.Module):

    """
    Transformers do not have understanding of positions. 
    So we add positions vector to it.
    (B,S,C) => (B,S,C)
    """

    def __init__(self, seq_len, embdim):
        super().__init__()

        # self.pe is a lookup matrix of shape (S, E), self.pe[i] represents ith seq
        self.pe = torch.zeros(seq_len, embdim) 



        self.pe[:, 0::2] = torch.sin(z)
        self.pe[:, 1::2] = torch.cos(z)

        # if you want we can unsqueeze to add B dim in self.pe if not it will broadcast
        # in Vanilla Transformer, Positional Encodings are not learnt
        self.pe = nn.Parameter(self.pe, requires_grad=True)
    
    def forward(self, x):
        # x shape=> (B,T,C)
        B,T,C = x.shape
        # only adding upto T (Seq len)
        x = x + self.pe[:T, :]
        return x


class LayerNormalization(nn.Module):

    """
    Layer Norm Implementation
    """
    def __init__(self, embdim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(embdim))
        self.beta = nn.Parameter(torch.zeros(embdim))
    
    def forward(self, x):
        xmean = x.mean(dim=-1, keepdim=True)
        xstd  = x.std(dim=-1, keepdim=True)
        out = self.alpha * ((x-xmean)/(xstd+1e-6)) + self.beta
        return out
