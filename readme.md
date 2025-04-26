# Custom GPT Model from Scratch

---

## Table of Contents

- [Project Overview](#project-overview)
- [GPT Architecture Overview](#gpt-architecture-overview)
- [Code Structure](#code-structure)
- [Training Details](#training-details)
- [Installation Requirements](#installation-requirements)
- [Training Time Estimation](#training-time-estimation)
- [Final Remarks](#final-remarks)

---

## Project Overview

The purpose of this project is to demonstrate a deep understanding of the GPT architecture by building it manually from scratch in PyTorch.  
The project covers everything from token embedding, position embedding, multi-head self-attention, feedforward networks, transformer blocks, and the final language modeling head, without using any external libraries like Huggingface.

An example notebook has also been provided which trains a ~10 million parameter GPT model from scratch and demonstrates text generation.

The same code can be used to build a model with any number of parameters just by altering architectural hyperparameters in config.py 

---

## GPT Architecture Overview

- **GPT is a stack of Transformer Decoders**: GPT models are built by stacking multiple transformer decoder blocks, each consisting of masked self-attention and feedforward layers.
- **Token Embedding**: Converts each input token into a continuous vector representation.
- **Position Embedding**: Injects positional information into the model since transformers have no inherent sense of order.
- **Self-Attention Mechanism**: Each token attends to all previous tokens in the sequence using scaled dot-product attention with causal masking.
- **Multi-Head Attention**: Multiple attention heads allow the model to focus on different parts of the sequence simultaneously.
- **Feedforward Networks**: A two-layer fully connected network applied at each position separately.
- **Residual Connections and Layer Normalization**: Applied after attention and feedforward blocks to improve training stability.
- **Stacked Transformer Blocks**: Multiple transformer layers are stacked to increase model capacity.
- **Language Modeling Head**: A final linear layer projects the output back to vocabulary size for prediction.

This structure closely follows the original GPT design principles.

---

## Code Structure

| File | Description |
|:---|:---|
| `config.py` | Contains all model hyperparameters like embedding size, number of heads, dropout rate, etc. |
| `model.py` | Defines the core model: attention heads, multi-head attention, transformer blocks, and the final model. |
| `data.py` | Implements data batching functions used during training and validation. |
| `train.py` | Contains the training loop, evaluation metrics, and model checkpointing logic. |
| `inference.py` | Contains the text generation function using temperature and top-k sampling. |
| `example.ipynb` | Notebook demonstrating full model training and evaluation on a small dataset. |

---

## Training Details

- **Dataset**: A toy character-level dataset has been used for demonstration purposes.
- **Model Size**: Approximately 10 million parameters.
- **Hardware**: T4 GPU (Colab environment).
- **Training Method**:  
  Each epoch processes only a single batch (no full sweep through all batches).
- **Total Epochs** = 12,000 epochs

---

## Installation Requirements

Install the required libraries using:

```bash
pip install torch tqdm numpy matplotlib
```

or install all dependencies at once from `requirements.txt`.

---

## Potential improvements have been stated in the example notebook.