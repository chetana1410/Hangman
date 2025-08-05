"""
Data generation module for Hangman game.

This module contains functions to generate training data by simulating Hangman games.
It creates masked word examples that can be used to train the model.
"""

import random
import string
import os

def save_train_valid_split(words, val_size=10000, train_file='training_set.txt', val_file='validation_set.txt'):
    """
    Shuffles the input list of words and splits it into a training set and validation set.
    Writes these sets to separate files.

    Args:
        words (list): List of words to be split and saved.
        val_size (int, optional): Number of words in the validation set. Defaults to 10000.
        train_file (str, optional): File path for saving the training set. Defaults to 'training_set.txt'.
        val_file (str, optional): File path for saving the validation set. Defaults to 'validation_set.txt'.
    """
    # Shuffle the words list to ensure randomness
    random.shuffle(words)

    # Split the words into validation and training sets
    val_words = words[:val_size]
    train_words = words[val_size:]

    # Write validation set to file
    with open(val_file, 'w') as f:
        for word in val_words:
            f.write(word + '\n')

    # Write training set to file
    with open(train_file, 'w') as f:
        for word in train_words:
            f.write(word + '\n')

    print(f"{len(val_words)} words written to {val_file}")
    print(f"{len(train_words)} words written to {train_file}")


def generate_guess(actual_word, guessed_letters):
    """
    Generates a guess for the Hangman game. The guess is determined by a weighted random selection:
    40% chance to choose a letter from the actual word that hasn't been guessed yet, and 60% chance to choose
    a lowercase letter from all possible letters not already guessed.

    Args:
        actual_word (str): The word whose letters are to be used for guessing.
        guessed_letters (list of str): A list of letters that have been guessed so far.

    Returns:
        str or None: A letter guess if available, otherwise returns None.
    """
    if random.random() < 0.4:
        # 40% chance to guess from actual characters in the word
        available_chars = [ch for ch in actual_word if ch not in guessed_letters]
        if available_chars:
            return random.choice(available_chars)
        else:
            return None
    # 60% chance to guess from all lowercase letters
    available_chars = [ch for ch in string.ascii_lowercase if ch not in guessed_letters]
    if available_chars:
        return random.choice(available_chars)
    return None


def play_hangman(word, output_file, max_attempts=6):
    """
    Simulates a single game of Hangman. The function guesses letters from the actual word and updates the state
    of the guessed letters and masked word until either all letters are guessed or the maximum number of attempts is reached.

    Args:
        word (str): The target word for the Hangman game.
        output_file (str): The file where the game progress will be logged.
        max_attempts (int, optional): The maximum number of incorrect guesses allowed before the game ends. Defaults to 6.
    """
    masked_word = ['#' for _ in word]
    guessed_letters = []
    attempts = 0
    correct_guesses = 0

    # Ensure that the first guess is from the actual word
    while attempts < max_attempts and correct_guesses == 0:
        first_guess = generate_guess(word, guessed_letters)
        guessed_letters.append(first_guess)
        
        # Update the masked word with the first correct guess
        if first_guess in word:
            correct_guesses += word.count(first_guess)
            for idx, char in enumerate(word):
                if char == first_guess:
                    masked_word[idx] = char                    
        else:
            attempts += 1
            
    # Simulate remaining guesses
    while attempts < max_attempts and correct_guesses < len(word):
        with open(output_file, 'a') as file:
            file.write(f"{''.join(masked_word)},{''.join(guessed_letters)},{word}\n")
        
        guess = generate_guess(word, guessed_letters)
        if guess is None:
            break
            
        guessed_letters.append(guess)

        if guess in word:
            correct_guesses += word.count(guess)
            for idx, char in enumerate(word):
                if char == guess:
                    masked_word[idx] = char
        else:
            attempts += 1

    return


def simulate_hangman_game(input_file='training_set.txt', output_file='masked_training_set.txt', max_attempts=6):
    """
    Reads words from an input file, simulates playing Hangman for each word, and writes the results to an output file.

    Args:
        input_file (str, optional): The file containing the words to be used in the simulations. Defaults to 'training_set.txt'.
        output_file (str, optional): The file where the simulated games' progress will be logged. Defaults to 'masked_training_set.txt'.
        max_attempts (int, optional): The maximum number of incorrect guesses allowed per game. Defaults to 6.
    """
    words = None
    with open(input_file, 'r') as file:
        words = [line.strip() for line in file.readlines()]

    # Clear the output file before starting the simulation
    open(output_file, 'w').close()
    for word in words:
        play_hangman(word, output_file, max_attempts)


def check_masked_characters(file_path='masked_training_set.txt', mask_char='#'):
    """
    Check if masked words in a file follow specific rules.

    Args:
        file_path (str, optional): Path to the file containing masked words. Defaults to 'masked_training_set.txt'.
        mask_char (str, optional): Character used to mask words. Defaults to '#'.

    Returns:
        None

    Description:
        This function reads a file line by line, where each line contains a masked word and its actual word, separated by a comma.
        It checks if the masked word has at least 1 and at most n-1 number of masks, where n is the length of the word.
        Additionally, it checks if any character that is masked in the masked word is still unmasked anywhere in the word.
        If any errors are found, they are printed to the console. Otherwise, a success message is printed.
    """
    
    with open(file_path, 'r') as f:
        lines = f.readlines()

    errors = []  # To keep track of lines with issues

    for line in lines:
        # Split each line into masked_word and actual_word
        masked_word, guesses, actual_word = line.strip().split(',')

        # Checking if the masked word has atleast 1 and atmost n-1 number of masks
        num_of_masks = masked_word.count('#')
        if num_of_masks == 0 or num_of_masks == len(masked_word):
            errors.append((masked_word, guesses, actual_word))
            continue
            
        # Check if any character that is masked in the masked_word is still unmasked in the actual_word
        for i, char in enumerate(masked_word):
            if char == mask_char:
                actual_char = actual_word[i]
                if actual_char in masked_word:
                    errors.append((masked_word, guesses, actual_word))
                    break
            elif char not in guesses:
                errors.append((masked_word, guesses, actual_word))
                break

    # Output results
    if errors:
        print("Errors found:")
        for error in errors:
            print(error)
    else:
        print("All masked words are valid.")


def prepare_training_data(dictionary_file='words_250000_train.txt', val_size=10000):
    """
    Prepare training data from dictionary file.
    
    This function reads words from the dictionary file, splits them into training and validation sets,
    and then generates masked training data by simulating Hangman games.
    
    Args:
        dictionary_file (str, optional): Path to the dictionary file. Defaults to 'words_250000_train.txt'.
        val_size (int, optional): Size of the validation set. Defaults to 10000.
    """
    print("Reading words from dictionary...")
    words = []
    with open(dictionary_file, 'r') as file:
        for word in file:
            word = word.strip()  # Remove newline characters
            words.append(word)
    
    print(f"Read {len(words)} words from dictionary.")
    
    # Create train/validation split
    print("Creating train/validation split...")
    save_train_valid_split(words, val_size=val_size)
    
    # Generate masked training data
    print("Generating masked training data...")
    simulate_hangman_game(input_file='training_set.txt', output_file='masked_training_set.txt')
    
    # Verify the quality of generated data
    print("Checking masked data quality...")
    check_masked_characters()
    
    print("Training data preparation complete!")
