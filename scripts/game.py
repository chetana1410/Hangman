import numpy as np
import random
import torch
from tqdm import tqdm
import time
from scripts.model import ComputeLossWithMask
from scripts.utils import run_train_iteration, save_checkpoint, load_checkpoint
from scripts.data_processing import create_words_batch, read_and_shuffle_data

class HangmanGame:
    """
    Class to simulate the Hangman game for both training and evaluation.
    """
    
    def __init__(self, player, write_path='masked_training_set_new.txt', verbose=False, training=False):
        """
        Initialize the HangmanGame with a player and configuration options.
        
        Args:
            player: The player that will make guesses
            write_path (str, optional): Path to write training data. Defaults to 'masked_training_set_new.txt'.
            verbose (bool, optional): Whether to print game details. Defaults to False.
            training (bool, optional): Whether to generate training data. Defaults to False.
        """
        self.player = player
        self.words_list = []
        self.verbose = verbose
        self.training = training 
        self.write_path = write_path

    def load_training_words(self, file_path='training_set.txt'):
        """
        Load training words from a file.
        
        Args:
            file_path (str, optional): Path to training words file. Defaults to 'training_set.txt'.
            
        Returns:
            list: List of words
        """
        words = None
        with open(file_path, 'r') as file:
            words = [line.strip() for line in file.readlines()]
    
        return words
    
    def load_test_words(self, file_path="validation_set.txt"):
        """
        Load test words from a text file.
        
        Args:
            file_path (str, optional): Path to test words file. Defaults to "validation_set.txt".
            
        Returns:
            list: List of words
        """
        words = None
        with open(file_path, 'r') as file:
            words = [line.strip() for line in file.readlines()]
        
        return words

    def generate_question_answer_pairs(self, words):
        """
        Generate masked question and corresponding answer word pairs.
        
        Args:
            words (list): List of words
            
        Yields:
            tuple: (masked_question, answer_word)
        """
        for word in words:
            answer_word = word.strip()
            question = '#' * len(answer_word)  # Mask entire answer with '#'
            yield question, answer_word

    def play(self, qa_pairs, correct_guesses, total_games, num_words):
        """
        Play the Hangman game for a set of question-answer pairs.
        
        Args:
            qa_pairs: Question-answer pairs iterator
            correct_guesses (int): Number of correct guesses so far
            total_games (int): Total number of games played so far
            num_words (int): Number of words to play
            
        Returns:
            tuple: (updated_correct_guesses, updated_total_games)
        """
        if self.training:
            q_a_set = set()

        # Loop through each question-answer pair
        for question, answer in tqdm(qa_pairs, desc="Processing Words", total=num_words):
            self.player.new_game()
            remaining_attempts = 6
            success_rate = correct_guesses / total_games if total_games > 0 else 0
            
            if self.verbose:
                print(f"{'='*20} Game {total_games + 1} {'='*20} Success Rate: {success_rate:.2f}")
                print('Question:', " ".join(question))

            total_loss = 0
            num_guesses = 0

            past_guesses = ''
            
            # Gameplay loop
            while '#' in question and remaining_attempts > 0:
                if question.count("#") < len(question):
                    num_guesses += 1
                    
                guess = self.player.guess(question, answer, trail_number=6-remaining_attempts)
                if guess != '_' and guess != '#' and guess != '&':
                    past_guesses += guess
                    
                updated_question = []
                
                # Update the question with the correct guessed characters
                for masked_char, answer_char in zip(question, answer):
                    if masked_char == '#':  # Masked character
                        if answer_char == guess:  # Correct guess
                            updated_question.append(answer_char)
                        else:
                            updated_question.append(masked_char)  # Keep masked if incorrect
                    else:
                        updated_question.append(masked_char)  # Retain previously revealed characters
                
                question = "".join(updated_question)

                # If we're in training mode and this is a good training example, add it
                if self.training and question.count('#') < len(question) and '#' in question:
                    q_a_set.add((question, past_guesses, answer))
                
                # If the guess is incorrect, decrement attempts
                if guess not in answer:
                    remaining_attempts -= 1
        
                if self.verbose:
                    print(f"Question: {' '.join(question)} | Guess: {guess} | Attempts Left: {remaining_attempts} | Answer: {answer}")
                
            # Check if the player successfully guessed the word
            if '#' not in question:
                correct_guesses += 1
                    
            total_games += 1

        # If in training mode, write the new training examples to file
        if self.training:
            with open(self.write_path, 'a') as file:
                for qa in q_a_set:
                    file.write(f"{qa[0]},{qa[1]},{qa[2]}\n")
        
        return correct_guesses, total_games
        
    def run(self):
        """
        Run the hangman game with the loaded test words and track performance.
        
        Returns:
            float: Success rate
        """
        if not self.training:
            test_words = self.load_test_words()
            np.random.shuffle(test_words)
            words = test_words[:1000]  # Use the first 1000 shuffled words
        else:
            words = self.load_training_words()
            np.random.shuffle(words)
    
        # Initialize counters
        correct_guesses = 0
        total_games = 0

        print(f"Total Games: {len(words)}")
        l = 0
        step = 10000
        r = l + step
        
        while total_games < len(words):
            curr_correct_guesses = 0
            curr_total_guesses = 0
            qa_pairs = self.generate_question_answer_pairs(words[l:r])
            curr_correct_guesses, curr_total_guesses = self.play(
                qa_pairs, curr_correct_guesses, curr_total_guesses, 
                num_words=len(words[l:r])
            )
            final_success_rate = curr_correct_guesses / curr_total_guesses
            print(f"{curr_correct_guesses} successes out of {curr_total_guesses} games, Success Rate: {final_success_rate:.4f}")
            total_games += curr_total_guesses
            correct_guesses += curr_correct_guesses
            l = r
            r += step
            r = min(r, len(words))
            
        return correct_guesses/total_games


def train_model(training_file_path='masked_training_set.txt', use_checkpoint=True, 
                saving_dir='model/', num_epochs=100, batch_size=256):
    """
    Train the Hangman model on the specified training data.
    
    Args:
        training_file_path (str, optional): Path to training data. Defaults to 'masked_training_set.txt'.
        use_checkpoint (bool, optional): Whether to use a checkpoint. Defaults to True.
        saving_dir (str, optional): Directory to save models. Defaults to 'model/'.
        num_epochs (int, optional): Number of epochs to train. Defaults to 100.
        batch_size (int, optional): Batch size. Defaults to 256.
        
    Returns:
        None
    """
    from scripts.utils import get_model, run_train_iteration
    from scripts.player import Player
    import os
    
    # Create saving directory if it doesn't exist
    if not os.path.isdir(saving_dir):
        os.mkdir(saving_dir)

    # Model parameters
    num_attn_heads = 4
    model_dim = 128
    dropout_prob = 0.1
    ff_hidden_dim = 512
    num_encoders = 4
    
    # File paths
    model_save_path = os.path.join(saving_dir, 'dev.checkpoint')
    
    # Initialize model
    model, model_opt, optimizer, vocab, vocab_size, device = get_model(
        num_attn_heads, model_dim, dropout_prob, ff_hidden_dim, num_encoders
    )
    
    # Load data
    train_data, valid_data = read_and_shuffle_data(training_file_path)
    
    # Load checkpoint if available
    model, model_opt, model_save_path, optimizer, saved_epoch, valid_scores_history = load_checkpoint(
        model, model_opt, model_save_path, optimizer, use_checkpoint, device
    )
    
    print(f"Validation Accuracy History: {valid_scores_history}")
    if len(valid_scores_history):
        print(f"Max Validation History: {max(valid_scores_history)}")
    
    # Training loop
    for epoch in range(saved_epoch, num_epochs):
        print('\n', "*" * 10, f"Epoch: {epoch}", "*" * 10, '\n')
        model.train()
        compute_loss = ComputeLossWithMask(
            model_output_generator=model.output_generator, 
            ce_loss_weight=1.0, 
            cos_sim_weight=1.0, 
            optimizer=model_opt
        )
        train_data_iterator = create_words_batch(
            lines=train_data, vocabulary=vocab, batch_size=batch_size, 
            shuffle=False, device=device
        )
        
        # Training variables
        train_iteration = cumm_loss = cumm_samples = total_samples = 0
        start = time.time()

        # Batch training
        for i, batch in enumerate(train_data_iterator):
            batch_loss = run_train_iteration(model, compute_loss, batch, vocab_size, device)
            cumm_loss += batch_loss.item()
            cumm_samples += batch.source_seq.shape[0]
            total_samples += batch.source_seq.shape[0]
    
            train_iteration += 1
    
            if train_iteration % 1000 == 0:
                time_elapsed = time.time() - start
                print(f'epoch {epoch}, iter {train_iteration}, avg. loss {cumm_loss / cumm_samples:.5f} time_elapsed {time_elapsed:.2f}sec')
                start = time.time()
                cumm_loss = cumm_samples = 0

        time_elapsed = time.time() - start
        print(f'epoch {epoch}, iter {train_iteration}, avg. loss {cumm_loss / cumm_samples:.5f} time_elapsed {time_elapsed:.2f}sec')
        
        if total_samples:
            print(f'Epoch: {epoch}, Examples used: {total_samples} ')
            print(f"Saving model after Epoch: {epoch}")
            save_checkpoint(model, epoch, model_opt, valid_scores_history, model_save_path)            
                
            print('Testing the model with a simulated hangman game')
            player = Player(model_save_path)
            game = HangmanGame(player)
            accuracy = game.run()
            valid_scores_history.append(accuracy)
            print(f'Validation accuracy: {accuracy*100} %')
            
            if len(valid_scores_history)==1 or accuracy == max(valid_scores_history):
                print("Saving current best model based on validation accuracy")
                save_checkpoint(model, epoch, model_opt, valid_scores_history, os.path.join(saving_dir, 'best_model.checkpoint'))


def self_play():
    """
    Perform self-play to generate more training data and improve the model.
    
    The function creates new training sets and trains the model on them until
    there's no significant improvement in accuracy.
    
    Returns:
        None
    """
    from scripts.player import Player
    import os
    
    i = 0
    prev_training_accuracy = 0
    curr_training_accuracy = 0
    
    while True:
        i += 1
        player = Player('model/best_model.checkpoint')
        game = HangmanGame(player, training=True, write_path=f'masked_training_set_new{i}.txt')
        print(f"Creating training set: masked_training_set_new{i}.txt")
        
        # Remove previous file if it exists
        try:
            os.remove(f'masked_training_set_new{i}.txt')
        except Exception as e:
            pass
            
        # Run the game to generate new training data
        accuracy = game.run()
        curr_training_accuracy = accuracy
        
        # Train model if there's a significant improvement
        if curr_training_accuracy - prev_training_accuracy > 0.01:
            print("Training model on new training data set")
            train_model(training_file_path=f'masked_training_set_new{i}.txt', num_epochs=25)
            prev_training_accuracy = curr_training_accuracy
        else:
            print("No significant increase in the training accuracy. Hence stopping self play.")
            break
