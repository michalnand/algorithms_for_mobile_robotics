import LibsControl

import numpy


import cv2


def scale_position(x, x_max, size):

    k = size/(2.0*x_max)
    q = size - k*x_max

    x = k*x + q

    x = numpy.clip(x, 0, size)

    return int(x)


class MPCRealTime:
    def __init__(self, dynamical_system, q, r, control_horizon, prediction_horizon, max_u, dt):

        self.dynamical_system = dynamical_system
        a_disc, b_disc, _     = LibsControl.c2d(self.dynamical_system.a, self.dynamical_system.b, self.dynamical_system.c, dt)

        #solve MPC controller
        self.mpc = LibsControl.MPC(a_disc, b_disc, q, r, control_horizon, prediction_horizon, max_u)
        self.u   = numpy.zeros((self.dynamical_system.b.shape[1], 1))

        self.x_trajectory = []

        self.energy = 0.0

    def step(self, xr):
        self.u = self.mpc.forward_trajectory(xr, self.dynamical_system.x, self.u)
    
        #compute plant output   
        x, y = self.dynamical_system.forward_state(self.u)

        self.energy+= (self.u**2).mean()

        d = ((xr[0, :] - x[:, 0])**2).mean()
        return d


    def render(self, img):
        x_max = 1.5
        
        x = self.dynamical_system.x[0, 0]
        y = self.dynamical_system.x[2, 0]

        x = scale_position(x, x_max, img.shape[0])
        y = scale_position(y, x_max, img.shape[1])
        img = cv2.circle(img, (x, y), 10, (1, 0, 0.3), -1)

        if len(self.x_trajectory) >= 500:
            self.x_trajectory.pop(0)
        self.x_trajectory.append(self.dynamical_system.x[:, 0])

        for n in range(len(self.x_trajectory) - 1):
            x0 = scale_position(self.x_trajectory[n][0], x_max, img.shape[0])
            y0 = scale_position(self.x_trajectory[n][2], x_max, img.shape[1])
            x1 = scale_position(self.x_trajectory[n+1][0], x_max, img.shape[0])
            y1 = scale_position(self.x_trajectory[n+1][2], x_max, img.shape[1])

            img = cv2.line(img, (x0, y0), (x1, y1), (1, 0, 0.3), 2)

        return img
    
    
