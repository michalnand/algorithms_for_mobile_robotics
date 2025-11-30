import numpy

class DifferentialDriveModel:

    def __init__(self, b0, b1, width, height):
        # system dynamics matrices
        self.A = numpy.array([
            [1, 0],
            [0, 1]
        ])

        self.B = numpy.array([
            [b0, 0],
            [0, b1]
        ])

        # robot dimensions
        self.width  = width
        self.height = height

        self.reset(0, 0, 0)
        
    def reset(self, initial_x, initial_y, initial_theta):
        # initialise state
        
        self.x = numpy.zeros((2, 1))

        self.x[1, 0] = initial_theta

        self.x_pos = initial_x
        self.y_pos = initial_y

        self.step(numpy.zeros((2, 1)))


    def step(self, u):
        # robot dynamics simulation
        x_new = self.A@self.x + self.B@u

        # calculate robot position in cartesian
        dpos    = x_new[0, 0] - self.x[0, 0]
        theta   = self.x[1, 0]  
        
        dx    = dpos*numpy.cos(theta)
        dy    = dpos*numpy.sin(theta)          

        self.x_pos+= dx
        self.y_pos+= dy

        # udpate new state
        self.x = numpy.array(x_new)

        # fill result values
        self.robot_state = {}
        self.robot_state["width"]  = self.width
        self.robot_state["height"] = self.height

        self.robot_state["x"]     = self.x
        self.robot_state["x_pos"] = self.x_pos
        self.robot_state["y_pos"] = self.y_pos
        self.robot_state["theta"] = theta

        return self.robot_state
    
    def get_state(self):
        return self.robot_state
