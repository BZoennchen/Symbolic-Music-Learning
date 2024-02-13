import torch.nn as nn
import torch
import torch.nn.functional as F


class Head(nn.Module):
    """ one head of self-attention """
    
    def __init__(self, n_embd, head_size, sequence_len, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(sequence_len, sequence_len)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # B, T, head_size
        q = self.query(x) # B, T, head_size
        _, _, head_size = q.shape #???
        
        wei = q @ k.transpose(-2, -1) * (head_size ** (-0.5)) # B, T, head_size @ B, head_size, T => B, T, T
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x) # B, T, head_size
        out = wei @ v # T, T @ B, T, head_size => B, T, head_size
        return out
        
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_heads, head_size, sequence_len, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, sequence_len, dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(head_size * n_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # concatenation of the results of each head
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    
class FeedForward(nn.Module):
    
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), 
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    
    def __init__(self, n_embd, n_heads, sequence_len, dropout):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_embd, n_heads, head_size, sequence_len, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # residual connection
        x = x + self.ffwd(self.ln2(x)) # residual connection
        return x
    
class TransformerDecoder(nn.Module):
    
    def __init__(self, vocab_size, sequence_len, n_embd, n_heads, n_blocks, dropout, device):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(sequence_len, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads, sequence_len, dropout) for _ in range(n_blocks)])
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = device
        
    def forward(self, idx):
        B, T = idx.shape
        
        token_emb = self.token_embedding_table(idx) # B, T, n_embd
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # T, n_embd        
        x = token_emb + pos_emb # B, T, n_embd + T, n_embd => B, T, n_embd
        x = self.blocks(x) # B, T, head_size
        logits = self.lm_head(x) # B, T, vocab_size
        return logits