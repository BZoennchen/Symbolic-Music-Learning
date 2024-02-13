import torch

##### start hyperparameters #####
sequence_len = 32
batch_size = 32
n_embd = 12 # has to be devisible by n_heads
n_heads = 2
n_blocks = 1
dropout = 0.2

criterion = torch.nn.CrossEntropyLoss()

learning_rate = 0.001
n_epochs = 10
eval_interval = 100
##### end hyperparameters #####