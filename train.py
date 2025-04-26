import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from model import FinalModel
from data import get_batch
from config import batch_size, lr, device, block_size, epochs, save_path
from tqdm.auto import tqdm
import os


def estimate_loss(model,train_data, val_data, batch_size, block_size):
    model.eval()
    
    inputs, targets = get_batch(train_data, val_data, batch_size, block_size, 'val')
    inputs, targets = inputs.to(device), targets.to(device)

    with torch.no_grad():
        logits, loss = model(inputs, targets)
        predictions = torch.argmax(logits, dim=-1)
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        correct = (predictions == targets).float().mean()
        perplexity = torch.exp(loss).item()

    return loss.item(), correct.item(), perplexity



def trainer(model, train_data, val_data, batch_size, lr, epochs, block_size, save_path):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    val_accuracies = []
    val_perplexities = []

    best_val_loss = float('inf')

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
      model.train()

      inputs, targets = get_batch(train_data, val_data, batch_size, block_size, 'train')
      inputs, targets = inputs.to(device), targets.to(device)

      optimizer.zero_grad()
      logits, loss = model(inputs, targets)
      loss.backward()
      optimizer.step()

      avg_train_loss = loss.item()
      val_loss, val_accuracy, val_perplexity = estimate_loss(model,train_data, val_data, batch_size, block_size)

      train_losses.append(avg_train_loss)
      val_losses.append(val_loss)
      val_accuracies.append(val_accuracy)
      val_perplexities.append(val_perplexity)
      if epoch%1000==0:
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.4f} | Val PPL: {val_perplexity:.4f}")

      if val_loss < best_val_loss:
          if not os.path.exists(save_path):
            os.makedirs(save_path)
          best_val_loss = val_loss
          torch.save(model.state_dict(), f"{save_path}/best_model.pth")

    plt.figure(figsize=(10, 5))
    plt.plot(range(epochs), train_losses, label='Train Loss')
    plt.plot(range(epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(epochs), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Epoch vs Accuracy')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(epochs), val_perplexities, label='Validation Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Epoch vs Perplexity')
    plt.legend()
    plt.show()
