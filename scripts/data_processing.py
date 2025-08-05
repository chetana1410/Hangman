import numpy as np
import torch
import random
from scripts.vocabulary import Vocabulary
from scripts.model import Batch

def read_and_shuffle_data(filepath, validation_filepath='validation_set.txt'):
    """
    Reads data from filepath and validation_filepath, shuffles them,
    and prints the lengths of the training and validation datasets.

    Args:
        filepath (string): File Path for training data.
        validation_filepath (string): File Path for validation data. Default is 'validation_set.txt'.
        
    Returns:
        tuple: A tuple containing two lists - train_data (list of words) and validation_data (list of words).
    """
    train_data = []
    with open(filepath, 'r') as file:
        for word in file:
            word = word.strip()  # Remove newline characters
            train_data.append(word)

    validation_data = []
    with open(validation_filepath, 'r') as file:
        for word in file:
            word = word.strip()  # Remove newline characters
            validation_data.append(word)

    for _ in range(4):
        random.shuffle(validation_data)
        random.shuffle(train_data)

    print(f"Length of training data: {len(train_data)}")
    print(f"Length of validation data: {len(validation_data)}")
    
    return train_data, validation_data


def pad_words(words, pad_token):
    """
    Pad list of words according to the longest word in the batch.

    Args:
        words (list[list[int]]): List of words, where each word is a list of characters.
        pad_token (int): Padding token.

    Returns:
        padded_words (list[list[int]]): List of padded words.
    """
    # Calculate the maximum sentence length
    max_word_length = max(len(word) for word in words)

    # Pad each sentence to the maximum length
    padded_words = [
        word + [pad_token] * (max_word_length - len(word))
        for word in words
    ]

    return padded_words


def convert_target_to_distribution(target, vocabulary, mask, device):
    """
    Convert target tensor to a distribution tensor.

    Args:
        target (Tensor): Target tensor.
        vocabulary (Vocabulary): Vocabulary object.
        mask (Tensor): Mask tensor.
        device (torch.device): Device to use.

    Returns:
        Tensor: Distribution tensor.
    """
    # Ensure the target tensor and mask are on the same device
    target = target.to(device)
    mask = mask.to(device)
    
    # Convert target tensor to numpy array (need to move to CPU first)
    target_numpy = (target * mask).cpu().numpy()
    
    # Add an extra column to the target numpy array
    extra_column = np.ones((target.shape[0], 1), dtype=target_numpy.dtype) * (vocabulary.char2id['z'] + 1)
    target_numpy = np.hstack((target_numpy, extra_column))
    
    # Calculate the distribution using bincount
    target_distribution = np.apply_along_axis(np.bincount, 1, target_numpy)[:, :-1]
    
    # Convert the distribution back to a tensor and move to the correct device
    target_distribution = torch.from_numpy(target_distribution).to(device)
    
    # Set the first element of the distribution to 0 as its padding token
    target_distribution[:, 0] = 0

    # Return the distribution tensor multiplied by the mask
    return target_distribution


def create_words_batch(lines, vocabulary, batch_size, device, shuffle=True):
    """
    Create a batch of word pairs from a list of lines.

    Args:
        lines (List[str]): List of lines.
        vocabulary (Vocabulary): Vocabulary object.
        batch_size (int): Batch size.
        device (Device): Device to use.
        shuffle (bool, optional): Whether to shuffle the lines. Defaults to True.

    Yields:
        Batch: A batch of word pairs.
    """
    if shuffle:
        np.random.shuffle(lines)

    source_buffer = []
    target_buffer = []
    mask_buffer = []
    guessed_buffer = []

    for line in lines:
        # Split the line into source and target words
        try:
            source_word, guessed_letters, target_word = line.split(',')
    
            # Check if the line is valid
            if len(line) > 1:
                # Convert the source and target words to     
                source_buffer.append([vocabulary.char2id[c] for c in source_word] + 
                                    [vocabulary.char2id['&']] + 
                                    [vocabulary.char2id[d] for d in guessed_letters])
                target_buffer.append([vocabulary.char2id[c] for c in target_word if c != '\n'])
                mask_buffer.append([c == '#' for c in source_word])
                guessed_buffer.append([vocabulary.char2id[c] for c in guessed_letters if c != '\n'])
                    
                # Check if the batch is full
                if len(source_buffer) == batch_size:
                    # Pad the source and target tensors
                    source_tensor = torch.tensor(pad_words(source_buffer, vocabulary.char2id['_']), device=device)
                    target_tensor = torch.tensor(pad_words(target_buffer, vocabulary.char2id['_']), device=device)
                    mask_tensor = torch.tensor(pad_words(mask_buffer, vocabulary.char2id['_']), device=device)
                    guessed_tensor = torch.tensor(pad_words(guessed_buffer, vocabulary.char2id['_']), device=device)
                    
                    # Calculate the target distribution
                    target_distribution = convert_target_to_distribution(target_tensor, vocabulary, mask_tensor, device=device)
                    target_distribution = torch.div(target_distribution, target_distribution.sum(dim=1)[:, None])
                    
                    # Create a batch object
                    batch = Batch(source_tensor, target_distribution, guessed_tensor, 
                                 mask_token=vocabulary.char2id['#'], pad_token=vocabulary.char2id['_'])
    
                    # Yield the batch
                    yield batch
    
                    # Reset the buffers
                    source_buffer = []
                    target_buffer = []
                    mask_buffer = []
                    guessed_buffer = []
        except:
            pass

    # Check if there are any remaining batches
    if len(source_buffer) != 0:
        # Pad the source and target tensors
        source_tensor = torch.tensor(pad_words(source_buffer, vocabulary.char2id['_']), device=device)
        target_tensor = torch.tensor(pad_words(target_buffer, vocabulary.char2id['_']), device=device)
        mask_tensor = torch.tensor(pad_words(mask_buffer, vocabulary.char2id['_']), device=device)
        guessed_tensor = torch.tensor(pad_words(guessed_buffer, vocabulary.char2id['_']), device=device)
        
        # Calculate the target distribution
        target_distribution = convert_target_to_distribution(target_tensor, vocabulary, mask_tensor, device=device)
        target_distribution = torch.div(target_distribution, target_distribution.sum(dim=1)[:, None])

        # Create a batch object
        batch = Batch(source_tensor, target_distribution, guessed_tensor, 
                     mask_token=vocabulary.char2id['#'], pad_token=vocabulary.char2id['_'])

        # Yield the batch
        yield batch
