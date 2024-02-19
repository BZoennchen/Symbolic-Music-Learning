import torch

##### start hyperparameters #####
sequence_len = 128*2
batch_size = 64
n_embd = 64 # has to be devisible by n_heads
n_heads = 4
n_blocks = 3
dropout = 0.2

criterion = torch.nn.CrossEntropyLoss()

learning_rate = 0.001
n_epochs = 10
eval_interval = 100
##### end hyperparameters #####