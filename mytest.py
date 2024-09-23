import os
import unittest
from time import time

from main import main

class MainTestCase(unittest.TestCase):
    def test_play(self):
        main(["play", "--n-rounds", "1","--my-agent=ML-monkey"])

if __name__ == '__main__':
    unittest.main()