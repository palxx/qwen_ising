from transformers import AutoTokenizer, AutoConfig
from qwen_model import TransformerBlock

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
config = AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")

text = "ising models are fun and I am training qwen on them"
inputs = tokenizer(text, return_tensors="pt")

print(inputs)
#print(config.vocab_size)

model = TransformerBlock(vocab_size=config.vocab_size)

x = model(inputs['input_ids'])   
        
print(x.shape)



















