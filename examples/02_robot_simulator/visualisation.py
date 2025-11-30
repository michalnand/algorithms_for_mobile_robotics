import numpy
import cv2

class Visualisation:

    def __init__(self, width = 512, height = 512, robot_min_x = 0.0, robot_min_y = 0.0, robot_max_x = 2.0, robot_max_y = 2.0):
        self.width  = width
        self.height = height

        self.robot_min_x = robot_min_x
        self.robot_min_y = robot_min_y
        self.robot_max_x = robot_max_x 
        self.robot_max_y = robot_max_y

        self.robots = []
        self.robots_path = []

    def add_robot(self, robot):
        self.robots.append(robot)
        self.robots_path.append([])


    def step(self):

        result_img = numpy.zeros((self.width, self.height, 3))

        for n in range(len(self.robots)): 
            state = self.robots[n].get_state()
            
            result_img = self._plot_robot(result_img, state, self.robots_path[n])

        cv2.imshow("visualisation", result_img)
        cv2.waitKey(1)


    def _normalise(self, x, y):

        k = self.width/(self.robot_max_x - self.robot_min_x)
        q = self.width - k*self.robot_max_x

        x_res = int(numpy.clip(k*x + q, 0, self.width))


        k = self.height/(self.robot_max_y - self.robot_min_y)
        q = self.height - k*self.robot_max_y

        y_res = (self.height-1) - int(numpy.clip(k*y + q, 0, self.height))

        return x_res, y_res
    
   


    def _plot_robot(self, result_image, state, trajectory, trajectory_length = 1024):
        x_pos   = state["x_pos"]
        y_pos   = state["y_pos"]
        theta   = state["theta"]

        width   = state["width"]
        height  = state["height"]

        color = numpy.array([1, 0, 0.2])

        if len(trajectory) > trajectory_length:
            trajectory.pop(0)   

        x_norm, y_norm = self._normalise(x_pos, y_pos)
        trajectory.append([x_norm, y_norm])

        pts = numpy.array(trajectory)

        result_image = cv2.polylines(result_image, [pts], False, 0.75*color, 1)

      

        x0  = x_pos - width/2
        y0  = y_pos - height/2
        x1  = x_pos + width/2
        y1  = y_pos + height/2

        theta = -theta*180.0/numpy.pi   

        x0, y0 = self._normalise(x0, y0)
        x1, y1 = self._normalise(x1, y1)

        center_x    = (x0 + x1) // 2
        center_y    = (y0 + y1) // 2   
        width       = abs(x1 - x0)
        height      = abs(y1 - y0)

        rect = ((center_x, center_y), (width, height), theta)

        # Get the 4 vertices of the rotated rectangle
        box = cv2.boxPoints(rect)
        box = numpy.array(box, dtype=int)
        
        # Draw the filled rotated rectangle
        result_image = cv2.fillPoly(result_image, [box], color)

        return result_image

