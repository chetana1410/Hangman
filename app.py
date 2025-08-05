import streamlit as st
import random
import string
import os
import glob
import torch
import time
from scripts.player import Player
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Set page config
st.set_page_config(
    page_title="GuessWars: AI vs Human",
    page_icon="⚔️",
    layout="centered"
)

# Define paths
MODEL_FOLDER = 'model'
DEFAULT_MODEL = 'model/best_model.checkpoint'
DICTIONARY_PATH = 'words_250000_train.txt'

# Default game settings
DEFAULT_MAX_ATTEMPTS = 6
DEFAULT_ROUNDS = 3
DEFAULT_WORDS = ['python', 'hangman', 'machine', 'learning', 'transformer', 'streamlit']

# Function to get available models
def get_available_models():
    try:
        models = glob.glob(f"{MODEL_FOLDER}/*.checkpoint")
        if not models:
            return [DEFAULT_MODEL]
        return models
    except Exception as e:
        st.warning(f"Error loading models: {e}")
        return [DEFAULT_MODEL]

# Function to draw battle state
def draw_battle_state(attempts, max_attempts=6, is_user_turn=False):
    # Create a blank image
    width, height = 300, 300
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Try to load a font, or use default
    try:
        font = ImageFont.truetype("Arial", 20)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw gallows
    draw.line([(50, 250), (250, 250)], fill='black', width=3)  # Base
    draw.line([(150, 250), (150, 50)], fill='black', width=3)  # Pole
    draw.line([(150, 50), (200, 50)], fill='black', width=3)  # Top
    draw.line([(200, 50), (200, 75)], fill='black', width=3)  # Rope
    
    # Draw man based on attempts
    if attempts >= 1:
        draw.ellipse([(185, 75), (215, 105)], outline='black', width=3)  # Head
    if attempts >= 2:
        draw.line([(200, 105), (200, 170)], fill='black', width=3)  # Body
    if attempts >= 3:
        draw.line([(200, 125), (170, 160)], fill='black', width=3)  # Left arm
    if attempts >= 4:
        draw.line([(200, 125), (230, 160)], fill='black', width=3)  # Right arm
    if attempts >= 5:
        draw.line([(200, 170), (170, 220)], fill='black', width=3)  # Left leg
    if attempts >= 6:
        draw.line([(200, 170), (230, 220)], fill='black', width=3)  # Right leg
        
    # Draw text
    player_text = "Your" if is_user_turn else "AI's"
    draw.text((50, 10), f"{player_text} incorrect attempts: {attempts}/{max_attempts}", fill='black', font=font)
    
    return image

# Function to initialize player
@st.cache_resource
def get_player(model_path):
    if os.path.exists(model_path):
        try:
            return Player(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.error(f"Model not found at {model_path}. Please train a model first.")
        return None

# Function to get random word
def get_random_word(dictionary_path):
    try:
        if os.path.exists(dictionary_path):
            with open(dictionary_path, 'r') as f:
                words = [line.strip() for line in f]
                return random.choice(words)
        else:
            return random.choice(DEFAULT_WORDS)
    except Exception as e:
        st.warning(f"Error loading dictionary: {e}")
        return random.choice(DEFAULT_WORDS)

# Function to handle user guessing
def user_guess_interface(word, masked_word, guessed_letters, incorrect_attempts, max_attempts):
    # Display the battle state image
    st.image(draw_battle_state(incorrect_attempts, max_attempts, True), caption="Battle State")
    
    # Display the current state
    st.subheader("Your Turn - Guess the Word:")
    word_display = ' '.join([c if c != '#' else '_' for c in masked_word])
    st.markdown(f"<h1 style='text-align: center;'>{word_display}</h1>", unsafe_allow_html=True)
    
    # Display guessed letters
    if guessed_letters:
        st.write(f"Letters you've guessed: {', '.join(guessed_letters)}")
    
    # Letter input
    alphabet = string.ascii_lowercase
    available_letters = [letter for letter in alphabet if letter not in guessed_letters]
    
    # Create rows of buttons for letter selection
    col_count = 7
    rows = [available_letters[i:i+col_count] for i in range(0, len(available_letters), col_count)]
    
    guess = None
    
    for row in rows:
        cols = st.columns(col_count)
        for i, letter in enumerate(row):
            with cols[i]:
                if st.button(letter.upper(), key=f"letter_{letter}"):
                    guess = letter
    
    return guess

# Main app
def main():
    st.title("GuessWars: AI vs Human")
    st.markdown("### The Ultimate Word Guessing Battle")
    
    # Sidebar
    st.sidebar.title("Game Settings")
    
    # Model selection dropdown
    available_models = get_available_models()
    default_index = 0
    for i, model in enumerate(available_models):
        if "best_model" in model:
            default_index = i
            break
    
    selected_model = st.sidebar.selectbox(
        "Select AI Model", 
        available_models,
        index=default_index,
        format_func=lambda x: os.path.basename(x)
    )
    
    # Max attempts slider
    max_attempts = st.sidebar.slider("Maximum Incorrect Attempts", 1, 10, DEFAULT_MAX_ATTEMPTS)
    
    # Number of rounds
    num_rounds = st.sidebar.slider("Number of Rounds", 1, 10, DEFAULT_ROUNDS)
    
    # Set time delay between AI guesses
    ai_guess_delay = st.sidebar.slider("AI Guess Delay (seconds)", 0.0, 2.0, 1.0, 0.1)
    
    # Initialize game state if not exists
    if 'game_active' not in st.session_state:
        st.session_state.game_active = False
    if 'current_round' not in st.session_state:
        st.session_state.current_round = 1
    if 'total_rounds' not in st.session_state:
        st.session_state.total_rounds = num_rounds
    if 'is_user_turn' not in st.session_state:
        st.session_state.is_user_turn = False
    if 'user_score' not in st.session_state:
        st.session_state.user_score = 0
    if 'ai_score' not in st.session_state:
        st.session_state.ai_score = 0
    if 'waiting_for_word' not in st.session_state:
        st.session_state.waiting_for_word = False
    if 'auto_guess' not in st.session_state:
        st.session_state.auto_guess = False
    
    # AI guessing state
    if 'ai_word' not in st.session_state:
        st.session_state.ai_word = ""
    if 'ai_masked_word' not in st.session_state:
        st.session_state.ai_masked_word = []
    if 'ai_guessed_letters' not in st.session_state:
        st.session_state.ai_guessed_letters = []
    if 'ai_incorrect_attempts' not in st.session_state:
        st.session_state.ai_incorrect_attempts = 0
    if 'ai_game_over' not in st.session_state:
        st.session_state.ai_game_over = False
    if 'ai_win' not in st.session_state:
        st.session_state.ai_win = False
    if 'ai_turn_round' not in st.session_state:
        st.session_state.ai_turn_round = 0
    
    # User guessing state
    if 'user_word' not in st.session_state:
        st.session_state.user_word = ""
    if 'user_masked_word' not in st.session_state:
        st.session_state.user_masked_word = []
    if 'user_guessed_letters' not in st.session_state:
        st.session_state.user_guessed_letters = []
    if 'user_incorrect_attempts' not in st.session_state:
        st.session_state.user_incorrect_attempts = 0
    if 'user_game_over' not in st.session_state:
        st.session_state.user_game_over = False
    if 'user_win' not in st.session_state:
        st.session_state.user_win = False
    
    # Function to start a new game
    def start_new_game():
        # Update rounds
        st.session_state.total_rounds = num_rounds
        st.session_state.current_round = 1
        
        # Reset scores
        st.session_state.user_score = 0
        st.session_state.ai_score = 0
        
        # Start with user entering a word
        st.session_state.waiting_for_word = True
        st.session_state.is_user_turn = False
        
        # Activate the game
        st.session_state.game_active = True
        
    # Function to start AI's turn with the provided word
    def start_ai_turn(word):
        # Initialize AI guessing state
        st.session_state.ai_word = word
        st.session_state.ai_masked_word = ['#'] * len(word)
        st.session_state.ai_guessed_letters = []
        st.session_state.ai_incorrect_attempts = 0
        st.session_state.ai_game_over = False
        st.session_state.ai_win = False
        st.session_state.ai_turn_round = 0
        st.session_state.waiting_for_word = False
        st.session_state.auto_guess = True
    
    # Function for AI to make a guess
    def ai_guess():
        player = get_player(selected_model)
        if player is None:
            st.error("Could not load the model. Please check if the model exists and is valid.")
            st.session_state.auto_guess = False
            return
        
        # Increment round
        st.session_state.ai_turn_round += 1
        
        # Get the current masked word
        masked_word = ''.join(st.session_state.ai_masked_word)
        
        # Make a guess
        guess = player.guess(masked_word, st.session_state.ai_word, trail_number=st.session_state.ai_incorrect_attempts)
        
        # Check if the guess is valid
        if not guess or guess in st.session_state.ai_guessed_letters:
            st.warning("AI made an invalid or repeated guess. Trying again...")
            return
        
        # Add to guessed letters
        st.session_state.ai_guessed_letters.append(guess)
        
        # Check if the guess is correct
        if guess in st.session_state.ai_word:
            # Update the masked word
            for i, char in enumerate(st.session_state.ai_word):
                if char == guess:
                    st.session_state.ai_masked_word[i] = char
            
            # Check if the word is completely guessed
            if '#' not in st.session_state.ai_masked_word:
                st.session_state.ai_game_over = True
                st.session_state.ai_win = True
                st.session_state.ai_score += 1
                st.session_state.auto_guess = False
        else:
            # Increment incorrect attempts
            st.session_state.ai_incorrect_attempts += 1
            
            # Check if the game is over
            if st.session_state.ai_incorrect_attempts >= max_attempts:
                st.session_state.ai_game_over = True
                st.session_state.auto_guess = False
    
    # Function to start user's turn
    def start_user_turn():
        # Get a random word for user to guess
        word = get_random_word(DICTIONARY_PATH)
        
        # Initialize user guessing state
        st.session_state.user_word = word
        st.session_state.user_masked_word = ['#'] * len(word)
        st.session_state.user_guessed_letters = []
        st.session_state.user_incorrect_attempts = 0
        st.session_state.user_game_over = False
        st.session_state.user_win = False
        st.session_state.is_user_turn = True
    
    # Function to process user's guess
    def process_user_guess(guess):
        # Add to guessed letters
        st.session_state.user_guessed_letters.append(guess)
        
        # Check if the guess is correct
        if guess in st.session_state.user_word:
            # Update the masked word
            for i, char in enumerate(st.session_state.user_word):
                if char == guess:
                    st.session_state.user_masked_word[i] = char
            
            # Check if the word is completely guessed
            if '#' not in st.session_state.user_masked_word:
                st.session_state.user_game_over = True
                st.session_state.user_win = True
                st.session_state.user_score += 1
        else:
            # Increment incorrect attempts
            st.session_state.user_incorrect_attempts += 1
            
            # Check if the game is over
            if st.session_state.user_incorrect_attempts >= max_attempts:
                st.session_state.user_game_over = True
    
    # Function to advance to next round
    def next_round():
        # Increment round
        st.session_state.current_round += 1
        
        # Reset to waiting for user to enter a word
        st.session_state.is_user_turn = False
        st.session_state.waiting_for_word = True
    
    # Game initialization
    if not st.session_state.game_active:
        st.write("### Game Rules")
        st.write("1. Each round consists of two parts: AI guesses your word, then you guess the AI's word.")
        st.write("2. One point is awarded for each successful guess.")
        st.write("3. After all rounds are completed, the winner is declared.")
        
        start_button = st.button("Start Game")
        if start_button:
            start_new_game()
            st.rerun()
    
    # Game active logic
    if st.session_state.game_active:
        # Display game stats in sidebar
        st.sidebar.subheader("Game Stats")
        st.sidebar.write(f"Round: {st.session_state.current_round}/{st.session_state.total_rounds}")
        st.sidebar.write(f"Your Score: {st.session_state.user_score}")
        st.sidebar.write(f"AI Score: {st.session_state.ai_score}")
        
        # Reset button
        if st.sidebar.button("Reset Game"):
            st.session_state.game_active = False
            st.rerun()
        
        # Display round information
        st.subheader(f"Round {st.session_state.current_round} of {st.session_state.total_rounds}")
        
        # User needs to enter a word for AI to guess
        if st.session_state.waiting_for_word:
            st.markdown("### Enter a word for the AI to guess")
            
            word_input = st.text_input("Enter a word:", key="current_word_input")
            
            submit_button = st.button("Submit Word")
            if submit_button:
                # Validate the word
                word = word_input.strip().lower()
                if not word or not all(c.isalpha() and c.islower() for c in word):
                    st.error("Please enter a valid word (lowercase letters only).")
                else:
                    start_ai_turn(word)
                    st.rerun()
        
        # AI's turn to guess
        elif not st.session_state.is_user_turn:
            st.markdown("### AI's Battle Turn")
            
            # Display the battle state image
            st.image(draw_battle_state(st.session_state.ai_incorrect_attempts, max_attempts), caption="Battle State")
            
            # Display the current state
            st.subheader("Current Word:")
            word_display = ' '.join([c if c != '#' else '_' for c in st.session_state.ai_masked_word])
            st.markdown(f"<h1 style='text-align: center;'>{word_display}</h1>", unsafe_allow_html=True)
            
            # Display guessed letters
            if st.session_state.ai_guessed_letters:
                st.write(f"AI guessed letters: {', '.join(st.session_state.ai_guessed_letters)}")
            
            # Auto-guess functionality
            if st.session_state.auto_guess:
                # Add a placeholder for the next guess button that will be auto-clicked
                placeholder = st.empty()
                with placeholder.container():
                    if st.button("AI Thinking...", key="auto_guess_button"):
                        ai_guess()
                        st.rerun()
                
                # Auto-click the button after delay
                time.sleep(ai_guess_delay)
                placeholder.empty()
                ai_guess()
                st.rerun()
            else:
                # Game controls when auto-guess is disabled or game is over
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col2:
                    if not st.session_state.ai_game_over:
                        # Manual guess button (for cases when auto-guess was disabled due to error)
                        guess_button = st.button("AI Make a Guess")
                        if guess_button:
                            ai_guess()
                            st.rerun()
                    else:
                        if st.session_state.ai_win:
                            st.success(f"The AI won! The word was: {st.session_state.ai_word}")
                        else:
                            st.error(f"The AI lost! The word was: {st.session_state.ai_word}")
                        
                        # Start user's turn
                        user_turn_button = st.button("Your Turn")
                        if user_turn_button:
                            start_user_turn()
                            st.rerun()
        
        # User's turn to guess
        else:
            st.markdown("### Your Battle Turn")
            
            if not st.session_state.user_game_over:
                # Get user's guess
                user_guess = user_guess_interface(
                    st.session_state.user_word,
                    st.session_state.user_masked_word,
                    st.session_state.user_guessed_letters,
                    st.session_state.user_incorrect_attempts,
                    max_attempts
                )
                
                if user_guess:
                    process_user_guess(user_guess)
                    st.rerun()
            else:
                # Display final state
                st.image(draw_battle_state(st.session_state.user_incorrect_attempts, max_attempts, True), caption="Battle State")
                
                word_display = ' '.join([c if c != '#' else '_' for c in st.session_state.user_masked_word])
                st.markdown(f"<h1 style='text-align: center;'>{word_display}</h1>", unsafe_allow_html=True)
                
                if st.session_state.user_win:
                    st.success(f"You won! You correctly guessed: {st.session_state.user_word}")
                else:
                    st.error(f"You lost! The word was: {st.session_state.user_word}")
                
                # Check if we need to go to next round or end game
                if st.session_state.current_round < st.session_state.total_rounds:
                    next_round_button = st.button("Next Round")
                    if next_round_button:
                        next_round()
                        st.rerun()
                else:
                    # Game complete - show final results
                    st.markdown("## Game Complete!")
                    st.markdown(f"### Final Score: You {st.session_state.user_score} - AI {st.session_state.ai_score}")
                    
                    if st.session_state.user_score > st.session_state.ai_score:
                        st.balloons()
                        st.markdown("### 🏆 You Win! 🏆")
                    elif st.session_state.ai_score > st.session_state.user_score:
                        st.markdown("### AI Wins!")
                    else:
                        st.markdown("### It's a Tie!")
                    
                    new_game_button = st.button("Play Again")
                    if new_game_button:
                        st.session_state.game_active = False
                        st.rerun()

# Run the app
if __name__ == "__main__":
    main()
