#!/usr/bin/env python3

import unittest
from pmlb import fetch_data

from brushgp import Program, Data, SearchSpace

class TestProgram(unittest.TestCase):
    def setup_class(self):
        X, y = fetch_data('breast_cancer_wisconsin', return_X_y=True)
        self.X = X
        self.y = y

    def test_data_does_initialize(self):
        self.data = Data(self.X, self.y)

    def test_can_create_search_space(self):
        self.search_space = SearchSpace(self.data)
    
    def test_does_create_program(self):
        self.program = Program(X, y)
        assert self.program is not None

    def test_does_fit_program(self):
        self.program.fit()

    def test