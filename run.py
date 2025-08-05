#!/usr/bin/env python3
"""
Hangman Game Runner

This is the main entry point to run the Hangman game. It imports and calls
the main function from the scripts package.
"""

import sys
import os
from scripts.main import main

if __name__ == "__main__":
    # Make sure the scripts directory is in the Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Call the main function
    main()
