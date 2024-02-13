import torch
import torch.nn.functional as F

import sys

from hyperparameters import learning_rate, eval_interval, sequence_len, batch_size, n_embd, n_heads, n_blocks, dropout
from transformer import TransformerDecoder
from preprocessor import StringToIntEncoder
from preprocessor import get_string_to_int
from tokenizer import TERM_SYMBOL

class Generator:
    def __init__(self, model, device, string_to_int: StringToIntEncoder):
        self.model = model
        self.device = device
        self.string_to_int = string_to_int
        
    def _next_event_number(self, idx, temperature:float):
        with torch.no_grad():
            outputs = self.model(idx[:,-sequence_len:])
            B, T, C = outputs.shape
            logits = outputs[:, -1, :]
            probs = F.softmax(logits / temperature, dim=1)  # B, C
            idx_next = torch.multinomial(probs, num_samples=1)
            return idx_next

    def _generate(self, seq: list[str]=None, max_len:int=None, temperature:float=1.0):
        with torch.no_grad():
            generated_encoded_song = []
            start_sequence = [self.string_to_int.encode(TERM_SYMBOL)]*sequence_len
            if seq != None:
                start_sequence = start_sequence + [self.string_to_int.encode(char) for char in seq]
                idx = torch.tensor([start_sequence], device=self.device)
                generated_encoded_song = seq.copy()
            else:
                idx = torch.tensor([start_sequence], device=self.device)
            
            while max_len == None or max_len > len(generated_encoded_song):
                idx_next = self.next_event_number(idx, temperature)
                char = self.string_to_int.decode(idx_next.item())
                if idx_next == self.string_to_int.encode(TERM_SYMBOL):
                    break
                idx = torch.cat((idx, idx_next), dim=1) # B, T+1, C
                generated_encoded_song.append(char)
                
            return generated_encoded_song

    def generate(self, temperature = 0.6, n_scores = 1, max_len=120):
        after_new_songs = []
        for _ in range(n_scores):
            encoded_song = self.generate(max_len=120,temperature=temperature)
            print(f'generated {" ".join(encoded_song)} conisting of {len(encoded_song)} notes')
            after_new_songs.append(encoded_song)

def main(model_path: str, vocab_size: int):
    
    # this is stupid to load all the data again!
    string_to_int = get_string_to_int()
    print(f'reload the data')
    vocab_size = len(string_to_int)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        #torch.backends.mps.empty_cache()
    else:
        device = torch.device('cpu')

    print(f'{device=}')
    
    model_path = './models/pretrained_32_2_2'
    model = TransformerDecoder(vocab_size, sequence_len, n_embd, n_heads, n_blocks, dropout)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    generator = Generator(model, device, string_to_int)

    n_scores = 5
    temperature = 0.6
    generator.generate(temperature=temperature, n_scores=n_scores)

if __name__ == '__main__':
    model_name = sys.argv[1]
    model_path = './models/'+model_name
    main(model_path)