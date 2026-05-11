import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbed(nn.Module):
    def __init__(self, vocab_size:int, embedding_dim:int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)


    def forward(self, input_ids):
        return self.embedding(input_ids)


class RotaryPositonalEmbed:
    def __init__(self, dim, theta = 10000):

        assert dim % 2 == 0, 'hidden dim should be divisible by 2'
        self.inv = 1/(theta**((2*torch.arange(0, dim, 2).float())/dim))


    def apply_rope(self, x):
        #print('inside apply rope', x.shape)
        batch, tokens, hiddem_dim, _ = x.shape
        assert hiddem_dim % 2 == 0, "should be zero"
        positions = torch.arange(tokens).float()
        angles = torch.einsum('s, d -> sd', positions, self.inv)
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(2)
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(2)
        #print('cos.shape', cos.shape)
        x_even = x[:,:,:, 0::2]
        x_odd = x[:,:,:, 1::2]
        #print('x_even.shape', x_even.shape)
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
        x_out = torch.empty_like(x)
        x_out[..., 0::2]= x_rot_even
        x_out[..., 1::2] = x_rot_odd
        return x_out


class MaskedGroupedQuery(nn.Module):
    def __init__(self, embeddings_size, num_q_heads, num_kv_heads, theta=10000):
        super().__init__()

        assert embeddings_size % num_q_heads == 0, "not matching q heads"
        assert num_q_heads % num_kv_heads == 0, "not matching kv heads"

        self.head_dim = embeddings_size // num_q_heads

        self.embeddings_size = embeddings_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_q_heads//num_kv_heads

        self.q = nn.Linear(embeddings_size, num_q_heads * self.head_dim, bias=False)
        self.k = nn.Linear(embeddings_size, num_kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(embeddings_size, num_kv_heads * self.head_dim, bias= False)
        self.o = nn.Linear(num_q_heads * self.head_dim, embeddings_size, bias=False)

        self.rope = RotaryPositonalEmbed(self.head_dim, theta)

    def forward(self, x):
        #print('inside MaskedGroupedQuery checking x shape', x.shape)
        batch, seq_len, _ = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        #print('checking q k v', q.shape)

        q = q.view(batch, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope.apply_rope(q)
        k = self.rope.apply_rope(k)

        k = k.repeat_interleave(4, dim=1)
        v = v.repeat_interleave(4, dim=1)

        atten_scores = torch.matmul(q, k.transpose(-2, -1))
        atten_scores = atten_scores/(self.head_dim ** 0.5)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=bool), diagonal=1)
        atten_scores = atten_scores.masked_fill(causal_mask, float('-inf'))
        atten_wei = F.softmax(atten_scores, dim=-1)        
        out = torch.matmul(atten_wei, v)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch, seq_len, self.embeddings_size)
        out = self.o(out)

        return out

class FeedForward(nn.Module):
    def __init__(self, embeddings_size, intermidiate_size):
        super().__init__()

        self.ll1 = nn.Linear(embeddings_size, intermidiate_size, bias=False)
        self.up_proj = nn.Linear(embeddings_size, intermidiate_size, bias=False)
        self.ll2 = nn.Linear(intermidiate_size, embeddings_size, bias=False)

    def forward(self, x):
        gate = self.ll1(x)
        up= self.up_proj(x) 
        x = F.silu(gate) * up
        x = self.ll2(x)

        return x



class block(nn.Module):
    def __init__(self, vocab_size, embeddings_size, num_q_heads, num_kv_heads, ff_size, theta=10000):
        super().__init__()
        self.embeddings_size= embeddings_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.ff_size = ff_size
        self.vocab_size = vocab_size
        self.theta = theta
        self.rms_norm1 = nn.RMSNorm(embeddings_size)
        self.masked_grouped_query = MaskedGroupedQuery(embeddings_size, num_q_heads, num_kv_heads, theta=10000)
        self.rms_norm2 = nn.RMSNorm(embeddings_size)
        self.ff = FeedForward(embeddings_size, ff_size)
        #self.final_ll = nn.Linear(embeddings_size, vocab_size)

    def forward(self, x):
        y = self.rms_norm1(x)
        #print('inside block y 1', y.shape)
        y = self.masked_grouped_query(y)
        #print('inside block y 2', y.shape)        
        y = y + x
        z = self.rms_norm2(y)
        z = self.ff(z)
        z = y + z
        return z
    

class TransformerBlock(nn.Module):
    def __init__(self, vocab_size, embeddings_size=2560, num_q_heads=32, num_kv_heads=8, ff_size=9728, num_blocks =36, theta=10000):
        super().__init__()
        self.num_blocks = num_blocks
        self.embeddings_size= embeddings_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.ff_size = ff_size
        self.vocab_size = vocab_size
        self.theta = theta
        self.token_embed = TokenEmbed(vocab_size, embeddings_size)
        self.blocks = nn.ModuleList([block(vocab_size, embeddings_size, num_q_heads, num_kv_heads, ff_size, theta=10000) for n in range(num_blocks)])
        self.final_rms = nn.RMSNorm(embeddings_size)
        self.final_ll = nn.Linear(embeddings_size, vocab_size)
        
    def forward(self, inputs, labels = None):

        x = self.token_embed(inputs)
        #print('inside transformer block x', x.shape)

        for block in self.blocks:
            x = block(x)

        x = self.final_rms(x)
        x = self.final_ll(x)

        if labels is not None:
            original_logits = x[:, :-1, :].contiguous()
            target_logits = labels[:,1:].contiguous()

            loss = F.cross_entropy(original_logits.view(-1, self.vocab_size), target_logits.view(-1))

        return {
            "logits": x ,
            "loss": loss
        }

        return x