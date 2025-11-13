from differential_drive_model import *
from visualisation            import *

import matplotlib.pyplot as plt


if __name__ == "__main__":
    # input parameters
    dt      = 0.001         #time step, 0.001 second
    r_wheel = 15.0*0.001    #wheel radius 15mm
    l_wheels= 95.0*0.001    #wheels brace distance, 95mm
    rpm_max = 750           #motor RPM speed max, 750

    width   = 2*l_wheels
    height  = l_wheels


    v_max = 2.0*numpy.pi*rpm_max*r_wheel/60.0
    w_max = 4.0*numpy.pi*rpm_max*r_wheel/(60.0*l_wheels)

    b0 = v_max*dt
    b1 = w_max*dt   
    
    # initialise robot
    robot = DifferentialDriveModel(b0, b1, width, height)
    robot.reset(0, 0, 0)

    # model validation - send robot to known trajectory

    # simulate triangular motion

    # at velocity v_max, time step dt, 
    # we estimate number of steps required to travel 1 meter
    n_steps_forward = round(1.0/(v_max*dt))

    # at turn rate w_max, time step dt, 
    # we estimate number of steps required to turn 120 degrees
    
    # degrees to radians
    angle = 120.0*numpy.pi/180.0 
    n_steps_turn = round(angle/(w_max*dt))

    x_pos = []
    y_pos = []

    for j in range(3):
        # move robot for 1m distance
        for n in range(n_steps_forward):
            u = numpy.array([[1.0],[0.0]])

            state = robot.step(u)

            x_pos.append(state["x_pos"])
            y_pos.append(state["y_pos"])

        # turn robot at 120 degrees
        for n in range(n_steps_turn):
            u = numpy.array([[0.0],[1.0]])

            state = robot.step(u)

            x_pos.append(state["x_pos"])
            y_pos.append(state["y_pos"])

    print("terminal state")
    print("x_pos = ", round(state["x_pos"], 4))
    print("y_pos = ", round(state["y_pos"], 4))
    print("theta = ", round(state["theta"]*180.0/numpy.pi, 4))
    
    
    plt.plot(x_pos, y_pos)
    plt.show()

    robot.reset(0.5, 0.5, 0)
    visualisation = Visualisation()

    visualisation.add_robot(robot)


    while True:
        # move robot for 1m distance
        for n in range(n_steps_forward):
            u = numpy.array([
                [1.0],
                [0.0]
            ])

            state = robot.step(u)
            
            if n%10 == 0:
                visualisation.step()


        # turn robot at 120 degrees
        for n in range(n_steps_turn):
            u = numpy.array([
                [0.0],
                [1.0]
            ])

            state = robot.step(u)

            if n%10 == 0:
                visualisation.step()

    
