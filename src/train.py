
from datetime import datetime


import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from hyperparameters import learning_rate, eval_interval, sequence_len, batch_size, n_embd, n_heads, n_blocks, dropout, n_epochs, criterion
from transformer import TransformerDecoder
from tokenizer import GridEncoder
from preprocessor import get_dataset


PATH_TO_MODELS = './../models/'

class Trainer:
    def __init__(self, model, device=torch.device('cpu')):
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.model = model
        self.device = device

    # Initializing in a separate cell so we can easily add more epochs to the same run
    def train(self, n_epochs: int, train_loader: DataLoader, val_loader: DataLoader, respect_val: bool = True):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        best_vloss = 1_000_000

        for epoch in range(n_epochs):    
            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch, writer, n_epochs, train_loader)
            
            self.model.train(False)
            running_vloss = 0.0
            
            for i, vdata in enumerate(val_loader):
                
                local_X, local_y = vdata
                local_X, local_y = local_X.to(self.device), local_y.to(self.device)
                            
                voutputs = self.model(local_X)
                
                B, T, C = voutputs.shape
                voutputs = voutputs.view(B*T, C)
                local_y = local_y.view(B*T)
                
                vloss = criterion(voutputs, local_y)
                running_vloss += vloss
                
            avg_vloss = running_vloss / (i+1)
            print(
                f'Epoch [{epoch+1}/{n_epochs}], Train-Loss: {avg_loss:.4f}, Val-Loss: {avg_vloss:.4f}')
            
            writer.add_scalars('Training vs. Validation Loss', {'Training': avg_loss, 'Validation': avg_vloss}, epoch)
            writer.flush()
            
            if not respect_val or (respect_val and avg_vloss < best_vloss):
                best_vloss = avg_vloss
                model_path = PATH_TO_MODELS+'/_model_{}_{}'.format(timestamp, epoch)
                torch.save(self.model.state_dict(), model_path)

    def train_one_epoch(self, epoch_index, tb_writer, n_epochs, train_loader: DataLoader):
        running_loss = 0.0
        last_loss = 0.0
        all_steps = n_epochs * len(train_loader)
        
        for i, data in enumerate(train_loader):
            local_X, local_y = data
            local_X, local_y = local_X.to(self.device), local_y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(local_X)
            
            #print(local_X.shape, local_y.shape)
            
            B, T, C = outputs.shape
            outputs = outputs.view(B*T, C)
            local_y = local_y.view(B*T)
            loss = criterion(outputs, local_y)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            if i % eval_interval == eval_interval-1:
                last_loss = running_loss / eval_interval  # loss per batch
                
                steps = epoch_index * len(train_loader) + (i+1)
                
                print(
                    f'Epoch [{epoch_index+1}/{n_epochs}], Step [{steps}/{all_steps}], Loss: {last_loss:.4f}')
                tb_x = epoch_index * len(train_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.
                
        return last_loss


def main():

    dataset, string_to_int = get_dataset()
    vocab_size = len(string_to_int)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=True)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        #torch.backends.mps.empty_cache()
    else:
        device = torch.device('cpu')
    print(f'device is {device}')
    
    model = TransformerDecoder(vocab_size, sequence_len, n_embd, n_heads, n_blocks, dropout, device=device)
    model.to(device)
    
    trainer = Trainer(model, device=device)
    trainer.train(n_epochs, train_loader=train_loader, val_loader=val_loader)


if __name__ == '__main__':
    main()