import torch
from config import file_name, train_split


def prepare_data():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(file_name, 'r', encoding='utf-8') as f:
        text = f.read()

    characters = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(characters)}
    itos = {i: ch for i, ch in enumerate(characters)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(train_split * len(data))
    train_data = data[:n].to(device)
    val_data = data[n:].to(device)

    return train_data, val_data, encode, decode,characters


def get_batch(train_data, val_data, batch_size, block_size, split):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)

    return x, y
