import torch
import numpy as np
from scripts.vocabulary import Vocabulary
from scripts.utils import get_most_probable_chars, guess_first_letter, model_init

class Player:
    """
    Player class for the Hangman game.
    
    This class is responsible for making guesses in the Hangman game. It uses a pre-trained
    model to predict the next letter to guess.
    """
    
    def __init__(self, model_path, num_attn_heads=4, model_dim=128, dropout_prob=0.1, ff_hidden_dim=512, num_encoders=4):
        """
        Initialize the Player with model parameters.
        
        Args:
            model_path (str): Path to the saved model checkpoint
            num_attn_heads (int, optional): Number of attention heads. Defaults to 4.
            model_dim (int, optional): Model dimension. Defaults to 128.
            dropout_prob (float, optional): Dropout probability. Defaults to 0.1.
            ff_hidden_dim (int, optional): Feed-forward hidden dimension. Defaults to 512.
            num_encoders (int, optional): Number of encoder layers. Defaults to 4.
        """
        self.vocabulary = Vocabulary()
        self.guessed_letters = []
        self.length_wise_most_probable_chars = get_most_probable_chars()
        self.device = self.check_device()
        self.num_attn_heads = num_attn_heads
        self.model_dim = model_dim
        self.dropout_prob = dropout_prob
        self.ff_hidden_dim = ff_hidden_dim
        self.num_encoders = num_encoders
        self.model = self.load_model(model_path)

    def check_device(self):
        """
        Check available device for model inference.
        
        Returns:
            torch.device: Device to use for inference
        """
        device = torch.device("cpu")
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        return device
        
    def guess(self, question, answer, trail_number=0):
        """
        Make a guess for the current state of the word.
        
        Args:
            question (str): Current state of the word with '#' for masked characters
            answer (str): The answer word (used in training mode)
            trail_number (int, optional): Number of previous attempts. Defaults to 0.
            
        Returns:
            str: The guessed character
        """
        # For the first guess, use the letter frequency statistics
        if question.count("#") == len(question):
            pred = guess_first_letter(self.length_wise_most_probable_chars, len(question), trail_number)
            self.guessed_letters.append(pred)
            return pred
        
        # For subsequent guesses, use the model
        guessed = [self.vocabulary.char2id[l] for l in self.guessed_letters]
        p = self.model_prediction(question, answer, ''.join(self.guessed_letters))
        p[guessed] = -np.inf  # Mask already guessed letters
        pred = self.vocabulary.id2char[np.argmax(p)]
        self.guessed_letters.append(pred)
        return pred
                
    def load_model(self, path):
        """
        Load the model from a checkpoint.
        
        Args:
            path (str): Path to the checkpoint file
            
        Returns:
            nn.Module: Loaded model
        """
        model, _, _ = model_init(self.num_attn_heads, self.model_dim, self.dropout_prob, self.ff_hidden_dim, self.num_encoders)
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = model.to(self.device)
        return model

    def new_game(self):
        """
        Reset the player for a new game.
        """
        self.guessed_letters = []
        
    def model_prediction(self, masked_word, answer, guesses):
        """
        Get model prediction for the next letter to guess.
        
        Args:
            masked_word (str): Current state of the word with '#' for masked characters
            answer (str): The answer word
            guesses (str): String of previously guessed letters
            
        Returns:
            numpy.ndarray: Probability distribution over the alphabet
        """
        with torch.no_grad():
            # Prepare input for the model
            src = torch.tensor([[self.vocabulary.char2id[c] for c in masked_word] + 
                               [self.vocabulary.char2id['&']] + 
                               [self.vocabulary.char2id[d] for d in guesses]], device=self.device)
            src_mask = ((src != self.vocabulary.char2id['#']) & (src != self.vocabulary.char2id['_'])).unsqueeze(-2)
            
            # Get model output
            out = self.model.forward(src, src_mask)
            generator_mask = torch.zeros(src.shape[0], len(self.vocabulary.char2id), device=self.device)
            generator_mask = generator_mask.scatter_(1, src, 1)
            p, _ = self.model.output_generator(out, generator_mask)
            p = p.squeeze(0)
            
            return p.cpu().detach().numpy()
