from transformers import AutoTokenizer, AutoConfig
import torch
import torch.nn as nn

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
config = AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")

text = "palavi is the best and learning what to do"
inputs = tokenizer(text, return_tensors="pt")

print(inputs)
# print(config)

class TokenEmbed(nn.Module):
    def __init__(self, vocab_size:int, embedding_dim:int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)


    def forward(self, input_ids):
        return self.embedding(input_ids)

tokenizer_embedding = TokenEmbed(tokenizer.vocab_size, embedding_dim=2560)
embeddings = tokenizer_embedding(inputs["input_ids"])
#print(embeddings.shape)

rms_norm = nn.RMSNorm([1, 10, 2560])
normalized_input = rms_norm(embeddings)
#print(normalized_input.shape)

class RotaryPositonalEmbed:
    def __init__(self, dim, theta = 10000):

        assert(dim % 2 == 0, 'hidden dim should be divisible by 2')
        self.inv = 1/(theta**((2*torch.arange(0, dim, 2).float())/dim))


    def apply_rope(self, x):
        batch, tokens, hiddem_dim = x.shape
        assert hiddem_dim % 2 == 0, "should be zero"
        positions = torch.arange(tokens).float()
        angles = torch.einsum('s d -> sd', positions, self.inv)
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(2)
        sin= torch.sin(angles).unsqueeze(0).unsqueeze(2)
        x_even = x[:,:,:, 0::2]
        x_odd = x[:,:,:, 1::2]
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

        self.rope = RotaryPositonalEmbed(embeddings_size, theta)

    def forward(self, x):
        batch, seq_len, hidden_size = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.view(batch, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope.apply_rope(q)
        k = self.rope.apply_rope(k)

        k = k.repeat_interleave(4, dim=1)
        v = v.repeat_interleave(4, dim=1)

        atten_scores = torch.matmul(q, k.transpose(-2, -1))
        atten_scores = atten_scores/(self.head_dim ** 0.5)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        atten_scores = atten_scores.masked_fill(causal_mask, float('-inf'))
        atten_wei = nn.softmax(atten_scores, dim=-1)
        
        out = torch.matmul(atten_wei, v)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch, seq_len, self.embeddings_size)
        out = self.o(out)

        return out

class FeedForward(nn.Module):
    def __init__(self, embeddings_size, intermidiate_size):

        self.ll1 = nn.Linear(embeddings_size, intermidiate_size, bias=False)
        self.up_proj = nn.Linear(embeddings_size, intermidiate_size, bias=False)
        self.ll2 = nn.Linear(intermidiate_size, embeddings_size, bias=False)

    def forward(self, x):
        gate = self.ll1(x)
        up= self.up_proj(x)
        x = nn.SiLU(gate) * up
        x = self.ll2(x)

        return x



class block(nn.Module):
    def __init__(self, embeddings, embeddings_size, num_q_heads, num_kv_heads, ff_size, vocab_size, theta=10000):
        super().init()
        self.embeddings = embeddings
        self.embeddings_size= embeddings_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.ff_size = ff_size
        self.vocab_size = vocab_size
        self.theta = theta
        self.rms_norm1 = nn.RMSNorm(embeddings)
        self.masked_grouped_query = MaskedGroupedQuery(embeddings_size, num_q_heads, num_kv_heads, theta=10000)
        self.rms_norm2 = nn.RMSNorm(embeddings)
        self.ff = FeedForward(embeddings_size, ff_size)
        #self.final_ll = nn.Linear(embeddings_size, vocab_size)

    def forward(self, x):
        y = self.rms_norm1(x)
        y = self.masked_grouped_query(self.embeddings_size, self.num_q_heads, self.num_kv_heads, theta=10000)
        y = y + x
        z = self.rms_norm2(y)
        z = FeedForward(z)
        z = y + z
        return z
    

class TransformerBlock:
    def __init__(self):
        pass
        
    def FeedForward(self):
        pass

    
        




















