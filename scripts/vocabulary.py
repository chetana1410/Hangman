import string

class Vocabulary:
    """A class to manage character vocabulary."""
    
    def __init__(self):
        """Initialize the vocabulary with special characters and alphabet."""
        self._special_chars = ['_', '#', '&'] 
        self._alphabet = string.ascii_lowercase
        self._char2id = self._create_char2id()
        self._id2char = {v: k for k, v in self._char2id.items()}

    def _create_char2id(self):
        """Create a dictionary mapping characters to unique IDs."""
        char2id = {}
        for char in self._special_chars:
            char2id[char] = len(char2id)
        for char in self._alphabet:
            char2id[char] = len(char2id)
        return char2id

    @property
    def char2id(self):
        """Return the character to ID dictionary."""
        return self._char2id

    @property
    def id2char(self):
        """Return the ID to character dictionary."""
        return self._id2char
