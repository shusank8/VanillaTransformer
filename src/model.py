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

class MultiHeadAttention(nn.Module):
    """
    Crux of transformer, Multi Head Attention
    Each token gets information from other token, 
    """

    def __init__(self, embdim, num_heads):
        self.embdim = embdim
        self.num_heads = num_heads
        self.query = nn.Linear(embdim, embdim)
        self.key = nn.Linear(embdim, embdim)
        self.value = nn.Linear(embdim, embdim)
        self.out = nn.Linear(embdim, embdim)
        assert embdim%num_heads==0, "Embdim: {embdim} must be divisible by Num_Heads: {num_heads}"
        self.num_heads = num_heads
        self.head_dim = embdim // num_heads
    
    @staticmethod
    def attention(q, k, v, mask):
        
        head_dim = q.shape[-1]
        # attention
        attention_scores = (q @ k.tranpose(-2, -1)) / (head_dim)**(1/2)

        attention_scores.masked_fill_(mask==0, float("-inf"))

        output = attention_scores @ v
        return output, attention_scores



    def forward(self, q, k, v, mask):
        """
        for encoder we only apply pad mask and for inital MHA we apply causal mask + pad mask
        i have applied & operator between causal mask and pad mask 
        resulting into a single mask

        for cross attention, q is from the target and k and v comes from source, mask also comes from source. 
        """

        # q,k,v size => (B,T,C)
        B,T,C = q.shape
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        # q,k,v size => (B,T,C)

        # resizing 
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim).tranpose(1,2)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_dim).tranpose(1,2)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim).tranpose(1,2)

        output, attention = MultiHeadAttention.attention(q, k, v, mask)

        # shape of output=> (B, NH, T, H)
        # shape of attention => (B, NH, SQ, SQ)
        # shape of output => (B,T,C)
        output = output.tranpose(1,2).contiguous().view(B,T,C)

        return self.out(output)
    

        
class FeedForward(nn.Module):

    """
    A FeedForward layer,
    In MHA, each token attends to another token so once we have infromation we use FeedForward to process those information
    """

    def __init__(self, embdim):
        super().__init__()
        self.ffd = nn.Sequential(
            nn.Linear(embdim, 4*embdim),
            nn.ReLU(),
            nn.Linear(4*embdim, embdim)
        )
    
    def forward(self, x):
        # shape of x=> (B,T,C)
        return self.ffd(x)
