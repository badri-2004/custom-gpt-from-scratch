import torch


batch_size = 32
lr = 1e-4
epochs = 20000
block_size = 256
n_embd = 384
heads = 6
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_path = "./models"
temperature = 1.0
top_k = 50
file_name = 'The Adventures of Sherlock Holmes.txt'
train_split = 0.9
start_char = '\n'
num_transformer_blocks = 6