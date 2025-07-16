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

        self.dropout = nn.Dropout(0.2)

        # self.pe is a lookup matrix of shape (S, E), self.pe[i] represents ith seq
        self.pe = torch.zeros(seq_len, embdim) 

        # positions is just the sequence of the position from 0,seq_len, shape => (SEQ_LEN, 1)
        positions = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        # SHAPE(256)
        emb_skip_dim = torch.arange(0, embdim, step=2, dtype=torch.float32)
        # (seqlen, 1) / (256) => (seqlen, 256)
        z = positions / (10000 ** (emb_skip_dim / embdim))

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
        return self.dropout(x)


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
        super().__init__()
        self.embdim = embdim
        self.num_heads = num_heads
        self.query = nn.Linear(embdim, embdim)
        self.key = nn.Linear(embdim, embdim)
        self.value = nn.Linear(embdim, embdim)
        self.out = nn.Linear(embdim, embdim)
        assert embdim%num_heads==0, "Embdim: {embdim} must be divisible by Num_Heads: {num_heads}"
        self.num_heads = num_heads
        self.head_dim = embdim // num_heads
        self.dropout = nn.Dropout(0.2)
    
    @staticmethod
    def attention(q, k, v, mask, dropout):
        
        head_dim = q.shape[-1]

            
        # attention
        attention_scores = (q @ k.transpose(-2, -1)) / (head_dim)**(1/2)

        if mask is not None:
            # for inference mask is None
            attention_scores.masked_fill_(mask==0, float("-inf"))

        attention_scores  = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

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
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim).transpose(1,2)

        output, attention = MultiHeadAttention.attention(q, k, v, mask, self.dropout)

        # shape of output=> (B, NH, T, H)
        # shape of attention => (B, NH, SQ, SQ)
        # shape of output => (B,T,C)
        output = output.transpose(1,2).contiguous().view(B,T,C)

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
            nn.Dropout(0.2),
            nn.Linear(4*embdim, embdim)
        )
    
    def forward(self, x):
        # shape of x=> (B,T,C)
        return self.ffd(x)


class EncoderBlock(nn.Module):
    """
    A single encoder block
    """
    def __init__(self, embdim, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(embdim, num_heads)
        self.layernorm1 = LayerNormalization(embdim)
        self.ffd = FeedForward(embdim)
        self.layernorm2 = LayerNormalization(embdim)
    
    def forward(self, x, mask):
        x = x + self.attn(x, x, x, mask)
        x = self.layernorm1(x)
        x = x + self.ffd(x)
        x = self.layernorm2(x)
        return x



class DecoderBlock(nn.Module):
    """
    A single decoder block
    """
    def __init__(self, embdim, num_heads):
        super().__init__()
        self.selfattn = MultiHeadAttention(embdim, num_heads)
        self.crossattn = MultiHeadAttention(embdim, num_heads)
        self.ffd = FeedForward(embdim)
        self.layernorm1 = LayerNormalization(embdim)
        self.layernorm2 = LayerNormalization(embdim)
        self.layernorm3 = LayerNormalization(embdim)
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        
        # adding x to itself for skip connection
        x = x + self.selfattn(x, x, x, tgt_mask)
        x = self.layernorm1(x)
        
        x = x + self.crossattn(x, encoder_output, encoder_output, src_mask)
        x = self.layernorm2(x)
        
        x = x + self.ffd(x)
        return self.layernorm3(x)


class Encoder(nn.Module):
    """
    Class to handle N Encoder Block
    """

    def __init__(self, layers):
        super().__init__()
        self.layers = layers
    
    def forward(self, x, mask):
        for layer in self.layers:
            x  = layer(x, mask)
        return x

class Decoder(nn.Module):
    """
    Class to handle N Decoder Block
    """

    def __init__(self, layers):
        super().__init__()
        self.layers = layers
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x  = layer(x, encoder_output, src_mask, tgt_mask)
        return x

class FinalProjection(nn.Module):
    """
    Final Projection to convert from embdim to vocab_size
    """

    def __init__(self, embdim, vocab_size):
        super().__init__()
        self.proj = nn.Linear(embdim, vocab_size)
    
    def forward(self, x):
        # shape of x => (B,T,C)
        # changing x to (B,T,VOCAB_SIZE)
        return self.proj(x)




class Transformer(nn.Module):
    """
    A class that builds everything
    """

    def __init__(self, encoder, decoder, src_emb, src_pos, tgt_emb, tgt_pos, finalproj):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_emb = src_emb
        self.src_pos = src_pos
        self.tgt_emb = tgt_emb
        self.tgt_pos = tgt_pos 
        self.finalproj = finalproj

    
    def encoder_fun(self, x, mask):
        x = self.src_emb(x)
        x = self.src_pos(x)
        return self.encoder(x, mask)

    def decoder_fun(self, x, encoder_output, src_mask, tgt_mask):
        x = self.tgt_emb(x)
        x = self.tgt_pos(x)
        return self.decoder(x, encoder_output, src_mask, tgt_mask)
    
    def projection(self, x):
        return self.finalproj(x)




def build_transformer(src_vocab_size, src_seq_len, tgt_vocab_size, tgt_seq_len, embdim, encoder_depth, decoder_depth, num_heads):
    src_emb = InputEmbeddings(src_vocab_size, embdim)
    src_pos = PositionalEmbeddings(src_seq_len, embdim)

    tgt_emb = InputEmbeddings(tgt_vocab_size, embdim)
    tgt_pos = PositionalEmbeddings(tgt_seq_len, embdim)

    encoder_blocks = []

    for _ in range(encoder_depth):
        encoder_block = EncoderBlock(embdim, num_heads)
        encoder_blocks.append(encoder_block)
    
    decoder_blocks = []

    for _ in range(decoder_depth):
        decoder_block = DecoderBlock(embdim, num_heads)
        decoder_blocks.append(decoder_block)
    
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    finalproj = FinalProjection(embdim, tgt_vocab_size)

    transformer = Transformer(
        encoder, decoder, src_emb, src_pos, tgt_emb, tgt_pos, finalproj
    )

    return transformer
