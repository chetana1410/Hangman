# GuessWars: AI vs Human

## Introduction

GuessWars is an AI-powered word guessing battle game based on the classic Hangman concept. In traditional Hangman, one player thinks of a word and the other player tries to guess it by suggesting letters. Each incorrect guess brings the player one step closer to losing (traditionally represented by drawing parts of a hanging man). The game ends when either the word is correctly guessed or too many incorrect guesses are made.

This project implements an advanced AI solution to the Hangman problem using machine learning. While humans typically use knowledge of letter frequencies and word patterns to make educated guesses, this solution employs a transformer-based neural network model that learns optimal guessing strategies from thousands of simulated games.

The system learns to:
1. Recognize patterns in partially revealed words
2. Consider previously guessed letters
3. Make strategic guesses that maximize the probability of winning
4. Adapt its strategy based on the current game state

This codebase provides a complete pipeline from data generation to model training and interactive gameplay, allowing you to train, test, and engage in word guessing battles against a sophisticated AI.

## Project Structure

The project is organized into the following structure:

- `scripts/`: Python modules implementing the game logic
  - `__init__.py`: Package initialization
  - `vocabulary.py`: Character vocabulary handling
  - `model.py`: Neural network model components
  - `data_processing.py`: Data loading and processing functions
  - `utils.py`: Utility functions for model training and evaluation
  - `player.py`: Player implementation for guessing letters
  - `game.py`: Game simulation and training
  - `main.py`: Main entry point for functionality
- `run.py`: Executable script to run the application
- `model/`: Directory containing model checkpoints
- Various text files with training and validation data

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- Requests
- tqdm
- Streamlit (for UI)
- Pillow (for image processing)

Install all required dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Game

The application can be run in two ways:

#### 1. Command Line Interface

Run the game using the `run.py` script with various command-line arguments:

```bash
# To prepare training data from the dictionary
./run.py --prepare-data [--dictionary-file PATH] [--val-size N]

# To train a model
./run.py --train [--training-file PATH] [--use-checkpoint] [--save-dir DIR] [--epochs N] [--batch-size N]

# To perform self-play to improve the model
./run.py --self-play

# To run a game simulation
./run.py --simulate [--model-path PATH] [--verbose]

# To play an interactive game (you provide the word, model guesses)
./run.py --play [--model-path PATH] [--max-attempts N]
```

### Command-Line Arguments

#### Data Preparation Options
- `--prepare-data`: Prepare training data from dictionary
- `--dictionary-file PATH`: Path to the dictionary file (default: 'words_250000_train.txt')
- `--val-size N`: Size of validation set (default: 10000)

#### General Options
- `--verbose`: Print detailed information during execution

#### Training Options
- `--train`: Train a new model
- `--training-file PATH`: Path to the training data file (default: 'masked_training_set.txt')
- `--use-checkpoint`: Use a checkpoint from the last run
- `--save-dir DIR`: Directory to save models (default: 'model/')
- `--epochs N`: Number of epochs to train (default: 100)
- `--batch-size N`: Batch size for training (default: 256)

#### Self-Play Option
- `--self-play`: Perform self-play to improve the model

#### Simulation Options
- `--simulate`: Run a game simulation
- `--model-path PATH`: Path to the model checkpoint (default: 'model/best_model.checkpoint')
- `--verbose`: Print detailed information during execution

#### Interactive Play Options
- `--play`: Play an interactive game where the model guesses a word
- `--model-path PATH`: Path to the model checkpoint (default: 'model/best_model.checkpoint')
- `--max-attempts N`: Maximum number of incorrect guesses allowed (default: 6)


## Implementation Details

### Data Preparation and Training Pipeline

The complete pipeline involves:

1. **Data Preparation**: Reading words from a dictionary file, splitting into training and validation sets, and generating masked training examples by simulating Hangman games.
2. **Model Training**: Using the masked examples to train a transformer-based model to predict the next letter to guess.
3. **Self-Play**: Improving the model through additional rounds of training based on games the model plays against itself.

### Model Architecture and Performance

The model is based on a transformer architecture that takes in the current state of the word (with masked characters) and previously guessed letters to predict the next best letter to guess.

For the first guess, the system uses letter frequency statistics derived from the training dictionary. For subsequent guesses, it uses the neural network model.

The model is trained using a combination of cross-entropy loss and cosine similarity.

**Performance**: The best model achieves a tested accuracy of 63% on unseen test data with a limit of 6 incorrect guesses. This means the AI can successfully guess the complete word before running out of attempts in nearly two-thirds of cases, significantly outperforming random guessing strategies.

## Improving the Model

The model can be improved by running the self-play option, which will generate more training data and train the model on it.

## Interactive Web UI: GuessWars Battle Arena

The project includes a Streamlit-based web interface for engaging in word guessing battles against the AI:

```bash
streamlit run app.py
```

The GuessWars Battle Arena features:

- Competitive gameplay where you and the AI take turns guessing each other's words
- Multi-round battles with score tracking to determine the ultimate champion
- Customizable AI models (select from available checkpoints)
- Adjustable difficulty settings and AI thinking speed
- Interactive battle visualization
- Virtual keyboard for letter selection
- Real-time battle statistics and performance tracking

Each battle consists of multiple rounds where you provide a word for the AI to guess, and then you try to guess the AI's word. Points are awarded for successful guesses, and after all rounds are completed, a winner is declared. Test your word-guessing prowess against an advanced neural network trained on thousands of games!
