#!/usr/bin/env python3

# Build a GP program in Brush via the Python interface.

import numpy as np

from unittest import TestCase

from brushgp import Program

class TestProgram(TestCase):
  def test_is_found(self):
    self.assertTrue(True)

  def test_makes_program(self):
    """Test whether brushgp can build a GP Program using a simple toy dataset.
    """
    # Data copied straight from test_program.cc
    X = np.array([[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0],
                  [2.0,1.0,6.0,4.0,5.0,8.0,7.0,5.0,9.0,10.0]], dtype=float)
    y = np.array([1.0,0.0,1.0,1.0,0.0,1.0,1.0,0.0,0.0,0.0], dtype=float)

    # Unlike in the C++ version, we don't need to explicitly construct a
    # search space.
    # TODO: Should we use a core set of default ops, or should the users be
    # able to define them individually? Or both...?
    prg = Program(X=x, y=y, depth=0, breadth=0, size=10)

  def test_back_prop(self):
    """Test whether brushgp can train a GP program using backprop and square
    error loss.
    """
    X = np.array([[0.85595296, 0.55417453, 0.8641915 , 0.99481109, 0.99123376],
                  [0.9742618 , 0.70894019, 0.94940306, 0.99748867, 0.54205151],
                  [0.5170537 , 0.8324005 , 0.50316305, 0.10173936, 0.13211973],
                  [0.2254195 , 0.70526861, 0.31406024, 0.07082619, 0.84034526]], dtype=float)
    y = np.array([3.55634251, 3.13854087, 3.55887523, 3.29462895, 3.33443517,
                  3.4378868 , 3.41092345, 3.5087468 , 3.25110243, 3.11382179], dtype=float)

    data = (X, y)
    
    prg = Program()

    prg.fit(data)

    y_pred = prg.predict(data)  # y_pred is a numpy array
    
    print()
    print("Calculating loss before training:")
    print("  y_pred: {0}".format(y_pred.T))
    print("  y:      {0}".format(y.T))
    print("  Loss:   {0}".format(prg.loss(y, y_pred)))  # Square error loss

    d_loss = prg.d_loss(y, y_pred)

    for x in range(20):
      print()
      print("Training epoch: {0}".format(x+1))
      prg.grad_descent(d_loss, data)
      print("  y_pred: {0}".format(y_pred.T))
      print("  y:      {0}".format(y.T))
      print("  Loss:   {0}".format(prg.loss(y, y_pred)))
      
      d_loss = prg.d_loss(y, y_pred)