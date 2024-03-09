import torch
import transformers
import json

result_dir = "./generated_text.json"

# GPT Neo
neo_large_model = transformers.GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
neo_large_tok = transformers.GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

# XLNet
# xlnet_model = transformers.XLNetLMHeadModel.from_pretrained("xlnet-large-cased")
# xlnet_tok = transformers.XLNetTokenizer.from_pretrained("xlnet-large-cased")

seed_texts = [
    """Report on Frankenstein:
    When reviewing the cultural inpact of Frankenstein it is clear that 
    """
]
sample_text = "When looking at related texts to frankenstein "
generator = neo_large_model
tokenizer = neo_large_tok

outputs = []

# temp = 0.1
max_len = 100
for temp in [0.1, 0.3, 0.5, .7, 0.9]:
    for seed_text in seed_texts:
        tokenized_input_text = tokenizer(sample_text, return_tensors="pt").input_ids
        input_len = len(tokenized_input_text)
        generated_tokens = generator.generate(tokenized_input_text, do_sample=True, temperature=temp,
                                              max_length=input_len + max_len)
        generated_text = tokenizer.batch_decode(generated_tokens)[0]

        outputs.append({
            'og_text': seed_text,
            'generated_text': generated_text,
            'temperature': temp
        })

with open(result_dir, 'w') as f:
    json.dump({'resuts': outputs}, f)
    f.close()
