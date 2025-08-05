#!/usr/bin/env python3
"""
Main entry point for the Hangman game application.

This script provides a unified interface to run the Hangman game using the modular code structure
that was refactored from the original notebook.
"""

import os
import argparse
import torch
import random
from scripts.player import Player
from scripts.game import HangmanGame, train_model, self_play
from scripts.data_generation import prepare_training_data

def check_model_path(model_path):
    """
    Check if the model exists at the specified path.
    
    Args:
        model_path (str): Path to the model checkpoint
        
    Returns:
        bool: True if model exists, False otherwise
    """
    return os.path.exists(model_path)

def run_game_simulation(model_path='model/best_model.checkpoint', verbose=True):
    """
    Run a simulation of the Hangman game.
    
    Args:
        model_path (str, optional): Path to the model checkpoint. Defaults to 'model/best_model.checkpoint'.
        verbose (bool, optional): Whether to print game details. Defaults to True.
    """
    # Check if model exists
    if not check_model_path(model_path):
        print(f"Model not found at {model_path}. Please train a model first.")
        return
    
    # Initialize player and game
    player = Player(model_path)
    game = HangmanGame(player, verbose=verbose)
    
    # Run the game
    accuracy = game.run()
    print(f"Game simulation complete. Success rate: {accuracy*100:.2f}%")


def play_interactive_game(model_path='model/best_model.checkpoint', max_attempts=6):
    """
    Run an interactive Hangman game where the user provides a word and the model tries to guess it.
    
    Args:
        model_path (str, optional): Path to the model checkpoint. Defaults to 'model/best_model.checkpoint'.
        max_attempts (int, optional): Maximum number of incorrect guesses allowed. Defaults to 6.
    """
    # Check if model exists
    if not check_model_path(model_path):
        print(f"Model not found at {model_path}. Please train a model first.")
        return
    
    # Get a word from the user
    word = input("Enter a word for the model to guess (or press enter for a random word): ").strip().lower()
    
    # If no word is provided, select a random word
    if not word:
        try:
            with open('words_250000_train.txt', 'r') as f:
                words = [line.strip() for line in f]
                word = random.choice(words)
        except Exception as e:
            print(f"Error selecting random word: {e}")
            print("Using a default word instead.")
            word = random.choice(['python', 'hangman', 'machine', 'learning', 'transformer'])
    
    # Validate the word (must contain only lowercase letters)
    if not all(c.isalpha() and c.islower() for c in word):
        print("The word must contain only lowercase letters. Using a default word instead.")
        word = random.choice(['python', 'hangman', 'machine', 'learning', 'transformer'])
    
    print(f"Word selected: {len(word) * '_'} ({len(word)} letters)")
    
    # Initialize the player with the model
    player = Player(model_path)
    
    # Initialize game state
    masked_word = ['#'] * len(word)
    guessed_letters = []
    incorrect_attempts = 0
    
    # Main game loop
    while '#' in masked_word and incorrect_attempts < max_attempts:
        # Display current state
        print(f"\nWord: {' '.join(masked_word)}")
        print(f"Guessed letters: {', '.join(guessed_letters) if guessed_letters else 'None'}")
        print(f"Incorrect attempts: {incorrect_attempts}/{max_attempts}")
        
        # Model makes a guess
        guess = player.guess(''.join(masked_word), word, trail_number=incorrect_attempts)
        print(f"Model guesses: {guess}")
        
        # Update guessed letters
        guessed_letters.append(guess)
        
        # Check if the guess is correct
        if guess in word:
            # Update the masked word
            for i, char in enumerate(word):
                if char == guess:
                    masked_word[i] = char
            print(f"Correct! '{guess}' is in the word.")
        else:
            incorrect_attempts += 1
            print(f"Wrong! '{guess}' is not in the word.")
    
    # Game over - display result
    print("\n" + "="*50)
    if '#' not in masked_word:
        print(f"The model WON! The word was: {word}")
        print(f"Incorrect attempts: {incorrect_attempts}/{max_attempts}")
    else:
        print(f"The model LOST! The word was: {word}")
        print(f"Incorrect attempts: {incorrect_attempts}/{max_attempts}")
        print(f"Final state: {' '.join(masked_word)}")


def main():
    """
    Main function to parse arguments and run the appropriate action.
    """
    parser = argparse.ArgumentParser(description='Hangman Game Runner')
    
    # Main action arguments
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--prepare-data', action='store_true', help='Prepare training data from dictionary')
    action_group.add_argument('--train', action='store_true', help='Train a new model')
    action_group.add_argument('--self-play', action='store_true', help='Perform self-play to improve the model')
    action_group.add_argument('--simulate', action='store_true', help='Run a game simulation')
    action_group.add_argument('--play', action='store_true', help='Play an interactive game where the model guesses a word')
    
    # Training arguments
    parser.add_argument('--training-file', type=str, default='masked_training_set.txt', 
                        help='Path to the training data file')
    parser.add_argument('--use-checkpoint', action='store_true', 
                        help='Use a checkpoint from the last run')
    parser.add_argument('--save-dir', type=str, default='model/', 
                        help='Directory to save models')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=256, 
                        help='Batch size for training')
    
    # Data preparation arguments
    parser.add_argument('--dictionary-file', type=str, default='words_250000_train.txt',
                      help='Path to the dictionary file for data preparation')
    parser.add_argument('--val-size', type=int, default=10000,
                      help='Size of validation set when preparing data')
    
    # Simulation arguments
    parser.add_argument('--model-path', type=str, default='model/best_model.checkpoint', 
                        help='Path to the model checkpoint')
    parser.add_argument('--verbose', action='store_true', 
                        help='Print detailed information during execution')
    parser.add_argument('--max-attempts', type=int, default=6,
                       help='Maximum number of incorrect guesses allowed (for interactive play)')
    
    
    args = parser.parse_args()
    
    # Execute the selected action
    if args.prepare_data:
        prepare_training_data(
            dictionary_file=args.dictionary_file,
            val_size=args.val_size
        )
    elif args.train:
        train_model(
            training_file_path=args.training_file,
            use_checkpoint=args.use_checkpoint,
            saving_dir=args.save_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size
        )
    elif args.self_play:
        self_play()
    elif args.simulate:
        run_game_simulation(
            model_path=args.model_path,
            verbose=args.verbose
        )
    elif args.play:
        play_interactive_game(
            model_path=args.model_path,
            max_attempts=args.max_attempts
        )

if __name__ == "__main__":
    main()
