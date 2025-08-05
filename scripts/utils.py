import string
import random
from collections import defaultdict
import torch
import torch.nn as nn
import os
import time
from scripts.vocabulary import Vocabulary
from scripts.model import Encoder, MultiHeadAttention, FeedForwardNetwork, TransformerEncoderLayer, SimpleGenerator, CombinedEmbedding, LearningRateScheduler
from torch.optim.adam import Adam

def get_training_words(filepath='words_250000_train.txt'):
    """
    Reads words from the specified file, keeping track of the maximum word length encountered.

    Args:
        filepath (str): Path to the file containing training words.

    Returns:
        tuple: A tuple containing two elements - a list of words and the maximum word length found in the file.
    """
    words = []
    max_word_length = 0
    with open(filepath, 'r') as file:
        for word in file:
            word = word.strip()  # Remove newline characters
            max_word_length = max(max_word_length, len(word))
            words.append(word)

    return words, max_word_length


def calculate_character_frequencies():
    """
    Calculate character frequencies for each word length.

    Returns:
        dict: Character frequencies for each word length
        int: Maximum word length in dictionary
    """
    words, max_word_length = get_training_words()
    word_length_char_freq = defaultdict(list)
    for length in range(max_word_length + 1):
        word_length_char_freq[length] = [0] * 26

    for word in words:
        for char in word:
            word_length_char_freq[len(word)][ord(char) - ord('a')] += 1

    return word_length_char_freq, max_word_length


def get_most_probable_chars():
    """
    Get the most probable characters for each word length.

    Returns:
        list: Most probable characters for each word length
    """
    word_length_char_freq, max_word_length = calculate_character_frequencies()
    length_wise_most_probable_chars = [None] * (max_word_length + 1)
    for length, chars in word_length_char_freq.items():
        most_prob_char = []
        chars_copy = chars.copy()  # Create a copy to avoid modifying the original
        for i in range(6):
            if any(chars_copy):  # Check if there are any non-zero values
                max_index = chars_copy.index(max(chars_copy))
                chars_copy[max_index] = 0
                most_prob_char.append(chr(max_index + ord('a')))
        length_wise_most_probable_chars[length] = most_prob_char

    return length_wise_most_probable_chars


def guess_first_letter(length_wise_most_probable_chars, word_length, trail_number):
    """
    Guess the first letter of a word based on character frequencies.

    Args:
        length_wise_most_probable_chars (dict): Length wise most probable chars
        word_length (int): Length of given word (question)
        trail_number (int): Trial number
         
    Returns:
        str: Guessed first letter
    """
    # Handle cases where trail_number is out of bounds or word_length has no data
    if not length_wise_most_probable_chars[word_length] or trail_number >= len(length_wise_most_probable_chars[word_length]):
        # Fall back to common vowels if we don't have specific data
        return random.choice(['e', 'a', 'i', 'o', 'u'])
    
    return length_wise_most_probable_chars[word_length][trail_number]


def model_init(num_attn_heads, model_dim, dropout_prob, ff_hidden_dim, num_encoders):
    """
    Initialize the model architecture.
    
    Args:
        num_attn_heads (int): Number of attention heads
        model_dim (int): Dimension of the model
        dropout_prob (float): Dropout probability
        ff_hidden_dim (int): Hidden dimension of feed-forward network
        num_encoders (int): Number of encoder layers
        
    Returns:
        tuple: (model, vocabulary, vocabulary_size)
    """
    vocab = Vocabulary()
    vocab_size = len(vocab.char2id)
    embedding = CombinedEmbedding(vocab_size=vocab_size, model_dim=model_dim, dropout_prob=dropout_prob, sep_token_id=vocab.char2id['&'])
    self_attn = MultiHeadAttention(num_heads=num_attn_heads, model_dim=model_dim, key_dim=model_dim // num_attn_heads, value_dim=model_dim // num_attn_heads, dropout_prob=dropout_prob)
    feed_forward = FeedForwardNetwork(model_dim=model_dim, ff_hidden_dim=ff_hidden_dim, dropout_rate=dropout_prob)
    encoder = TransformerEncoderLayer(self_attention_module=self_attn, feed_forward_module=feed_forward, hidden_size=model_dim, dropout_probability=dropout_prob)
    generator = SimpleGenerator(hidden_size=model_dim, vocabulary_size=vocab_size)
    model = Encoder(encoder_layer=encoder, embedding_layer=embedding, output_generator=generator, num_layers=num_encoders)
    return model, vocab, vocab_size


def initialize_model_parameters(model):
    """
    Initialize model parameters using Xavier uniform initialization.
    
    Args:
        model (nn.Module): Model to initialize
    """
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


def get_optimizers(model, model_dim, device, lr=0, betas=(0.9, 0.98), eps=1e-9, scaling_factor=1.3, warmup_steps=3000):
    """
    Create optimizers for model training.
    
    Args:
        model (nn.Module): Model to optimize
        model_dim (int): Dimension of the model
        device (torch.device): Device to use
        lr (float): Learning rate
        betas (tuple): Adam betas
        eps (float): Adam epsilon
        scaling_factor (float): Scaling factor for learning rate
        warmup_steps (int): Number of warmup steps
        
    Returns:
        tuple: (optimizer, model_opt)
    """
    optimizer = Adam(model.parameters(), lr=lr, betas=betas, eps=eps)
    model_opt = LearningRateScheduler(model_size=model_dim, scaling_factor=scaling_factor, warmup_steps=warmup_steps, optimizer=optimizer)
    return optimizer, model_opt


def get_model(num_attn_heads, model_dim, dropout_prob, ff_hidden_dim, num_encoders):
    """
    Create and initialize a new model.
    
    Args:
        num_attn_heads (int): Number of attention heads
        model_dim (int): Dimension of the model
        dropout_prob (float): Dropout probability
        ff_hidden_dim (int): Hidden dimension of feed-forward network
        num_encoders (int): Number of encoder layers
        
    Returns:
        tuple: (model, model_opt, optimizer, vocab, vocab_size, device)
    """
    model, vocab, vocab_size = model_init(num_attn_heads, model_dim, dropout_prob, ff_hidden_dim, num_encoders)
    initialize_model_parameters(model)
    device = torch.device("cpu")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    print("Model on Device:", device)
    model = model.to(device)
    optimizer, model_opt = get_optimizers(model, model_dim, device)
    return model, model_opt, optimizer, vocab, vocab_size, device


def load_checkpoint(model, model_opt, model_save_path, optimizer, use_checkpoint, device):
    """
    Load a checkpoint from a file if available. If no checkpoint is found or
    `use_checkpoint` is False, the function starts with fresh weights for the model.

    Args:
        model (torch.nn.Module): The neural network model to which the checkpoint will be loaded.
        model_opt (object): An object containing optimization options that might need updating from the checkpoint.
        model_save_path (str): Path to the file where the checkpoint is saved.
        optimizer (torch.optim.Optimizer or None): The optimizer used for training, if available. If not provided, set it to None.
        use_checkpoint (bool): A flag indicating whether to use a checkpoint from the last run.
        device (str): The device on which to load the model ('cpu' or 'cuda' or 'mps').

    Returns:
        tuple: A tuple containing the loaded model, updated optimizer object if provided, original save path,
               the same optimizer, the epoch number of the saved checkpoint, and the validation scores history.
    """
    try:
        if use_checkpoint:
            checkpoint = torch.load(model_save_path)
            saved_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            if optimizer:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            step = checkpoint['current_step']
            rate = checkpoint['current_rate']
            if model_opt:
                model_opt.current_step = step
                model_opt.current_rate = rate
            valid_scores_history = checkpoint['valid_scores_history']
            print(f'Reading checkpoint from epoch {saved_epoch}')
        else:
            saved_epoch = 0
            valid_scores_history = []
    except Exception as e:
        print(e)
        print("Some error occurred while loading saved model. Starting with fresh weights!")
        saved_epoch = 0
        valid_scores_history = []

    return model, model_opt, model_save_path, optimizer, saved_epoch, valid_scores_history


def save_checkpoint(model, epoch, model_opt, valid_scores_history, model_save_path):
    """
    Save a checkpoint containing the state of the model at a specific epoch. The checkpoint includes
    the model's state dictionary, optimizer state if available, and other relevant training information.

    Args:
        model (torch.nn.Module): The neural network model whose weights are to be saved.
        epoch (int): The current training epoch number for which the checkpoint is being created.
        model_opt (object or None): An object containing optimization options, including optimizer state if available.
                                   If not provided, set it to None.
        valid_scores_history (list): A list of validation scores that track performance over different epochs.
        model_save_path (str): The path where the checkpoint will be saved.
    """
    if model_opt:
        torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model_opt.optimizer.state_dict(),
            'current_rate': model_opt.current_rate,
            'current_step': model_opt.current_step,
            'valid_scores_history': valid_scores_history,
        }, model_save_path)
    else:
        torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': None,
            'current_rate': None,
            'current_step': None,
            'valid_scores_history': valid_scores_history,
        }, model_save_path)


def run_train_iteration(model, compute_loss, batch, vocab_size, device):
    """
    Perform a single training iteration on the given batch of data. This involves passing the source sequence
    through the model and computing the loss based on the target sequence using the provided loss function.

    Args:
        model (torch.nn.Module): The neural network model that will be used for prediction.
        compute_loss (function): A callable function to compute the loss between predicted and actual outputs.
        batch (object): An object containing the source sequence, target sequence, and other necessary data
                        for training, typically including masks and vocab size.
        vocab_size (int): The size of the vocabulary used by the model, required for creating a generator mask.
        device (str): The device on which to perform computations ('cpu' or 'cuda' or 'mps').

    Returns:
        torch.Tensor: The computed loss value for the batch.
    """
    generator_mask = torch.zeros(batch.source_seq.shape[0], vocab_size, device=device)
    generator_mask = generator_mask.scatter_(1, batch.source_seq, 1)
    guess_mask = torch.zeros(batch.source_seq.shape[0], vocab_size, device=device)
    guess_mask = guess_mask.scatter_(1, batch.guessed_seq, 1)
    out = model.forward(batch.source_seq, batch.source_mask)
    batch_loss = compute_loss(out, batch.target_seq, generator_mask, guess_mask)
    return batch_loss
