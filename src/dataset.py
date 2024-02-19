import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from tokenizer import TERM_SYMBOL

class ScoreDataset(Dataset):
    def __init__(self, enc_songs, stoi_encoder, sequence_len=64, transform=None, in_between=False, term_symbol=TERM_SYMBOL):
        self.stoi_encoder = stoi_encoder
        self.in_between = in_between
        self.transform = transform
        self.term_symbol = term_symbol
        self.X, self.y  = self.__generate_sequences(sequence_len, enc_songs)
        #self.tensors = (F.one_hot(torch.tensor(
        #    sequences), num_classes=len(self.stoi_encoder)).float(), torch.tensor(symbols))

    def __generate_sequences(self, sequence_len, enc_songs):
        X = []
        y = []
        for enc_song in enc_songs:
            chs = [TERM_SYMBOL]*sequence_len + enc_song + [TERM_SYMBOL]*sequence_len
            for i in range(len(enc_song)+1):
                ch1 = chs[i:sequence_len+i]
                X.append(list(map(lambda s: self.stoi_encoder.encode(s), ch1)))
                if self.in_between:
                    ch2 = chs[i+1:sequence_len+i+1]
                    y.append(list(map(lambda s: self.stoi_encoder.encode(s), ch2)))
                else:
                    ch2 = chs[sequence_len+i]
                    y.append(self.stoi_encoder.encode(ch2))
        return X, y
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X, y = torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
        return X, y