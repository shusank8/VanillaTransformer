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
        self.embeddings = nn.Embedding(vocab_size, embdim)
    
    def forward(self, x):
        return self.embeddings(x)
