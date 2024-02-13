import zipfile
import music21 as m21
import os
from pathlib import Path

from dataset import ScoreDataset
from tokenizer import Encoder
from tokenizer import GridEncoder, StringToIntEncoder

from hyperparameters import sequence_len

PATH_TO_ERK = './../data/erk.zip'

def kern_files_to_scores(dataset_path: str) -> list[m21.stream.Score]:
    # go through all the files in the ds and load them with music21
    scores = []
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            if str(file).endswith('.krn'):
                score = m21.converter.parse(os.path.join(path, file))
                scores.append(score)
    print(f'transformed {len(scores)} files to Score objects')
    return scores


def extract_and_load_songs_in_kern(dataset_path: str = PATH_TO_ERK):
    # Entpacke die zip-Datei, welche die Trainingsdaten enthält in den richtigen Ordner.
    pathname, extension = os.path.splitext(dataset_path)
    path_to_erk = Path(pathname).absolute()
    
    print(f'export files in {dataset_path} to {path_to_erk}')
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        dir = path_to_erk.parent.absolute()
        zip_ref.extractall(dir)
        
    scores = kern_files_to_scores(str(path_to_erk))
    return scores

def get_string_to_int(encoder: Encoder):
    print(f'one timestep represents {encoder.time_step} beats')
    
    scores = extract_and_load_songs_in_kern()
    enc_songs, invalid_song_indices = encoder.encode_songs(scores)
    print(f'there are {len(enc_songs)} valid songs and {len(invalid_song_indices)} songs')

    return StringToIntEncoder(enc_songs=enc_songs)

def get_dataset(encoder: Encoder = GridEncoder()) -> tuple[ScoreDataset, StringToIntEncoder]:
    print(f'one timestep represents {encoder.time_step} beats')
    
    scores = extract_and_load_songs_in_kern()
    enc_songs, invalid_song_indices = encoder.encode_songs(scores)
    print(f'there are {len(enc_songs)} valid songs and {len(invalid_song_indices)} songs')

    string_to_int = StringToIntEncoder(enc_songs=enc_songs)
    print(f'number of unique symbols: {len(string_to_int)}')
    
    dataset = ScoreDataset(enc_songs=enc_songs, stoi_encoder=string_to_int, sequence_len=sequence_len, in_between=True)
    return dataset, string_to_int