import json
import requests
import random
import time
import string
import secrets
import re
import collections
import torch
import numpy as np
from urllib.parse import parse_qs, urlencode, urlparse
import urllib3
from scripts.vocabulary import Vocabulary
from scripts.utils import get_most_probable_chars, guess_first_letter, model_init

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class HangmanAPIError(Exception):
    """
    Custom exception for Hangman API errors.
    """
    def __init__(self, result):
        self.result = result
        self.code = None
        try:
            self.type = result["error_code"]
        except (KeyError, TypeError):
            self.type = ""

        try:
            self.message = result["error_description"]
        except (KeyError, TypeError):
            try:
                self.message = result["error"]["message"]
                self.code = result["error"].get("code")
                if not self.type:
                    self.type = result["error"].get("type", "")
            except (KeyError, TypeError):
                try:
                    self.message = result["error_msg"]
                except (KeyError, TypeError):
                    self.message = result

        Exception.__init__(self, self.message)


class HangmanAPI:
    """
    API client for interacting with the Hangman game server.
    """
    
    def __init__(self, access_token=None, session=None, timeout=None, model_path='model/best_model.checkpoint'):
        """
        Initialize the HangmanAPI client.
        
        Args:
            access_token (str, optional): Access token for the API. Defaults to None.
            session (requests.Session, optional): Session object. Defaults to None.
            timeout (int, optional): Request timeout. Defaults to None.
            model_path (str, optional): Path to the model checkpoint. Defaults to 'model/best_model.checkpoint'.
        """
        self.hangman_url = self.determine_hangman_url()
        self.access_token = access_token
        self.session = session or requests.Session()
        self.timeout = timeout
        self.guessed_letters = []
        self.length_wise_most_probable_chars = get_most_probable_chars()
        
        # Dictionary setup
        full_dictionary_location = "words_250000_train.txt"
        self.full_dictionary = self.build_dictionary(full_dictionary_location)        
        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()
        
        # Model parameters
        self.num_attn_heads = 4
        self.model_dim = 128
        self.dropout_prob = 0.1
        self.ff_hidden_dim = 512
        self.num_encoders = 4
        
        # Device and model initialization
        self.device = self.check_device()
        self.vocabulary = Vocabulary()
        self.current_dictionary = []
        self.model = self.load_model(model_path)
        
    def check_device(self):
        """
        Check the available device for inference.
        
        Returns:
            torch.device: Device to use for inference
        """
        device = torch.device("cpu")
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        return device
        
    @staticmethod
    def determine_hangman_url():
        """
        Determine the fastest Hangman server URL.
        
        Returns:
            str: The URL of the fastest Hangman server
        """
        links = ['https://trexsim.com', 'https://sg.trexsim.com']

        data = {link: 0 for link in links}

        for link in links:
            requests.get(link)

            for i in range(10):
                s = time.time()
                requests.get(link)
                data[link] = time.time() - s

        link = sorted(data.items(), key=lambda x: x[1])[0][0]
        link += '/trexsim/hangman'
        return link

    def load_model(self, path):
        """
        Load the model from a checkpoint.
        
        Args:
            path (str): Path to the model checkpoint
            
        Returns:
            nn.Module: The loaded model
        """
        model, _, _ = model_init(self.num_attn_heads, self.model_dim, self.dropout_prob, self.ff_hidden_dim, self.num_encoders)
        try:
            checkpoint = torch.load(path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            model = model.to(self.device)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using fallback approach for letter guessing")
        return model
        
    def model_prediction(self, masked_word, guesses):
        """
        Get model prediction for the next letter to guess.
        
        Args:
            masked_word (str): The masked word with '#' for masked characters
            guesses (str): String of previously guessed letters
            
        Returns:
            numpy.ndarray: Probability distribution over the alphabet
        """
        try:
            with torch.no_grad():
                src = torch.tensor([[self.vocabulary.char2id[c] for c in masked_word] + 
                                  [self.vocabulary.char2id['&']] + 
                                  [self.vocabulary.char2id[d] for d in guesses]], device=self.device)
                src_mask = ((src != self.vocabulary.char2id['#']) & (src != self.vocabulary.char2id['_'])).unsqueeze(-2)
                out = self.model.forward(src, src_mask)
                generator_mask = torch.zeros(src.shape[0], len(self.vocabulary.char2id), device=self.device)
                generator_mask = generator_mask.scatter_(1, src, 1)
                p, _ = self.model.output_generator(out, generator_mask)
                p = p.squeeze(0)
                # Convert to float64 array to avoid type issues
                return p.cpu().detach().numpy().astype(np.float64)
        except Exception as e:
            # Fallback to letter frequency-based guessing
            print(f"Error in model prediction: {e}")
            char_freq = collections.Counter("".join(self.full_dictionary)).most_common()
            # Create a float array for frequencies
            p = np.zeros(len(self.vocabulary.char2id), dtype=np.float64)
            for char, freq in char_freq:
                if char in self.vocabulary.char2id:
                    p[self.vocabulary.char2id[char]] = float(freq)
            return p
                
    def guess(self, word):
        """
        Make a guess for the given word state.
        
        Args:
            word (str): Current state of the word (e.g., "_ p p _ e ")
            
        Returns:
            str: The guessed letter
        """
        # Clean the word: strip spaces and replace "_" with "#"
        question = word[::2].replace("_", "#")
        
        # First guess based on letter frequency by word length
        if question.count("#") == len(question):
            try:
                pred = guess_first_letter(self.length_wise_most_probable_chars, len(question), len(self.guessed_letters))
            except:
                # Fallback to vowels
                pred = random.choice(['a', 'e', 'i', 'o', 'u'])
                
            self.guessed_letters.append(pred)
            return pred
    
        # Subsequent guesses using the model
        guessed = [self.vocabulary.char2id[l] for l in self.guessed_letters]
        p = self.model_prediction(question, ''.join(self.guessed_letters))
        
        # Create a mask for already guessed letters
        valid_indices = np.ones_like(p, dtype=bool)
        for g in guessed:
            if g < len(valid_indices):
                valid_indices[g] = False
        
        # Apply mask by setting masked values to minimum
        masked_p = np.where(valid_indices, p, p.min() - 1)
        
        # Get prediction
        pred = self.vocabulary.id2char[np.argmax(masked_p)]
        self.guessed_letters.append(pred)

        return pred
       
    def build_dictionary(self, dictionary_file_location):
        """
        Build a dictionary from a file.
        
        Args:
            dictionary_file_location (str): Path to the dictionary file
            
        Returns:
            list: List of words from the dictionary
        """
        text_file = open(dictionary_file_location, "r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary
                
    def start_game(self, practice=True, verbose=True):
        """
        Start a new Hangman game.
        
        Args:
            practice (bool, optional): Whether this is a practice game. Defaults to True.
            verbose (bool, optional): Whether to print game details. Defaults to True.
            
        Returns:
            bool: True if the game was won, False otherwise
        """
        # Reset guessed letters and current dictionary
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary
                         
        response = self.request("/new_game", {"practice": practice})
        if response.get('status') == "approved":
            game_id = response.get('game_id')
            word = response.get('word')
            tries_remains = response.get('tries_remains')
            if verbose:
                print(f"Successfully start a new game! Game ID: {game_id}. # of tries remaining: {tries_remains}. Word: {word}.")
            
            # Ensure tries_remains is a valid integer
            try:
                tries_remain_int = int(tries_remains) if tries_remains is not None else 0
            except (ValueError, TypeError):
                tries_remain_int = 0
                
            while tries_remain_int > 0:
                # Get guessed letter from user code
                guess_letter = self.guess(word)
                    
                # Append guessed letter to guessed letters field
                self.guessed_letters.append(guess_letter)
                if verbose:
                    print(f"Guessing letter: {guess_letter}")
                    
                try:    
                    res = self.request("/guess_letter", {
                        "request": "guess_letter", 
                        "game_id": game_id, 
                        "letter": guess_letter
                    })
                except HangmanAPIError:
                    print('HangmanAPIError exception caught on request.')
                    continue
                except Exception as e:
                    print('Other exception caught on request.')
                    raise e
               
                if verbose:
                    print(f"Server response: {res}")
                
                status = res.get('status')
                # Safely get and convert tries_remains
                try:
                    tries_remains = res.get('tries_remains', 0)
                    tries_remain_int = int(tries_remains) if tries_remains is not None else 0
                except (ValueError, TypeError):
                    tries_remain_int = 0
                
                if status == "success":
                    if verbose:
                        print(f"Successfully finished game: {game_id}")
                    return True
                elif status == "failed":
                    reason = res.get('reason', '# of tries exceeded!')
                    if verbose:
                        print(f"Failed game: {game_id}. Because of: {reason}")
                    return False
                elif status == "ongoing":
                    word = res.get('word', '')  # Default to empty string if word is None
        else:
            if verbose:
                print("Failed to start a new game")
        return status == "success"
        
    def my_status(self):
        """
        Get the current status of the player.
        
        Returns:
            list: [total_practice_runs, total_recorded_runs, total_recorded_successes, total_practice_successes]
        """
        return self.request("/my_status", {})
    
    def request(self, path, args=None, post_args=None, method=None):
        """
        Make a request to the Hangman API.
        
        Args:
            path (str): API endpoint path
            args (dict, optional): Query parameters. Defaults to None.
            post_args (dict, optional): POST data. Defaults to None.
            method (str, optional): HTTP method. Defaults to None.
            
        Returns:
            dict: Response from the API
            
        Raises:
            HangmanAPIError: If the API returns an error
        """
        if args is None:
            args = dict()
        if post_args is not None:
            method = "POST"

        # Add access_token to post_args or args if needed
        if self.access_token:
            if post_args and "access_token" not in post_args:
                post_args["access_token"] = self.access_token
            elif "access_token" not in args:
                args["access_token"] = self.access_token

        # Rate limiting
        time.sleep(0.2)

        # Retry mechanism for API requests
        num_retry, time_sleep = 50, 2
        for it in range(num_retry):
            try:
                response = self.session.request(
                    method or "GET",
                    self.hangman_url + path,
                    timeout=self.timeout,
                    params=args,
                    data=post_args,
                    verify=False
                )
                break
            except requests.HTTPError as e:
                try:
                    # For requests.HTTPError, the content is in e.response.text
                    response_content = e.response.text
                    response = json.loads(response_content)
                except (AttributeError, json.JSONDecodeError):
                    response = {"error": str(e)}
                raise HangmanAPIError(response)
            except requests.exceptions.SSLError as e:
                if it + 1 == num_retry:
                    raise
                time.sleep(time_sleep)

        # Process response
        headers = response.headers
        if 'json' in headers['content-type']:
            result = response.json()
        elif "access_token" in parse_qs(response.text):
            query_str = parse_qs(response.text)
            if "access_token" in query_str:
                result = {"access_token": query_str["access_token"][0]}
                if "expires" in query_str:
                    result["expires"] = query_str["expires"][0]
            else:
                raise HangmanAPIError(response.json())
        else:
            raise HangmanAPIError('Maintype was not text, or querystring')

        # Check for API errors
        if result and isinstance(result, dict) and result.get("error"):
            raise HangmanAPIError(result)
            
        return result
