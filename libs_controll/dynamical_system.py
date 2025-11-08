import numpy
from .ode_solver import *

'''
continuous linear dynamical system

dx = Ax + Bu
y  = Cx

for solving runge kutta (RK4) is used

A matrix, shape (n_states, n_states)
B matrix, shape (n_states, n_inputs)
C matrix, shape (n_outputs, n_states)

x, system state, shape (n_states, 1)
u, controll input, shape (n_inputs, 1)
y, plant output, shape (n_outputs, 1)

default C is diagonal matrix, providing full state x observation
'''
class DynamicalSystem:

    def __init__(self, a, b, c, dt):

        self.a = a.copy()
        self.b = b.copy()

        if c is not None:
            self.c = c.copy()
        else:
            self.c = numpy.eye(self.a.shape[0])

        self.dt = dt

    #set initial state, used with forward_state
    def reset(self, x_initial = None):

        if x_initial is not None:
            self.x = x_initial.copy()
        else:
            self.x = numpy.zeros((self.a.shape[0], 1))

    
    #state-less forward func
    def forward(self, x, u):
        x_new, y = ODESolverRK4(self._step, x, u, self.dt)

        return x_new, y
    
    #state forward
    def forward_state(self, u):
        self.x, y = self.forward(self.x, u)
        return self.x, y
    
    #step callback for solver
    def _step(self, x, u):
        dx = self.a@x + self.b@u
        y  = self.c@x

        return dx, y