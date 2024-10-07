from lqr_real_time import *
from mpc_real_time import *

from sphere_model import *

def get_xr(n, horizon = 32, straight_length = 500):
    result   = []

    for i in range(horizon):
        idx_type   = (n+i)//straight_length
        idx_offset = (n+i)%straight_length

        if idx_type%4 == 0:
            x = [2.0*idx_offset/straight_length - 1.0, 0, -1.0, 0]
            result.append(x)
        elif idx_type%4 == 1:
            x = [1.0, 0, 2.0*idx_offset/straight_length - 1.0, 0]
            result.append(x)
        elif idx_type%4 == 2:
            x = [-(2.0*idx_offset/straight_length - 1.0), 0, 1.0, 0]
            result.append(x)
        else:
            x = [-1.0, 0, -(2.0*idx_offset/straight_length - 1.0), 0]
            result.append(x)


    result = numpy.array(result)
    return result

import time

if __name__ == "__main__":
    # create two identical dynamical systems
    ds_lqr = SphereDS()
    ds_mpc = SphereDS()

    # maximal action value
    max_u  = 0.5

    # solve LQR controller
    q = numpy.array([ [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]] )
    #r = numpy.array( [ [10.0, 0.0], [0.0, 10.0] ]) 
    r = numpy.array( [ [10**-3, 0.0], [0.0, 10**-3] ]) 

    xr_trajectory = get_xr(0, 1)

    #initial system state
    ds_lqr.reset()
    ds_lqr.x[:, 0] = xr_trajectory[0, :]

    lqr_loop = LQRRealTime(ds_lqr, q, r, max_u, ds_mpc.dt)


    
    #solve MPC controller
    control_horizon    = 8
    prediction_horizon = 64
    q = numpy.array([ [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]] )
    r = numpy.array( [ [10**-3, 0.0], [0.0, 10**-3] ]) 

    xr_trajectory = get_xr(0, prediction_horizon)

    #initial system state
    ds_mpc.reset()
    ds_mpc.x[:, 0] = xr_trajectory[0, :]

    mpc_loop = MPCRealTime(ds_mpc, q, r, control_horizon, prediction_horizon, max_u, ds_mpc.dt)


   

    lqr_idx = 0
    mpc_idx = 0

    #time.sleep(5)

    n = 0
    while True:
        xr = get_xr(lqr_idx, prediction_horizon)

        d = lqr_loop.step(xr[0, :])
        if d < 0.1:
            lqr_idx+= 1

        xr = get_xr(mpc_idx, prediction_horizon)

        d = mpc_loop.step(xr)
        if d < 0.1:
            mpc_idx+= 1

        if n%10 == 0:

            img = numpy.zeros((512, 512, 3), dtype=numpy.float32)

            xr = get_xr(n, 2000)

            for i in range(0, len(xr) - 1, 2):
                x0 = scale_position(xr[i][0], 1.5, img.shape[0])
                y0 = scale_position(xr[i][2], 1.5, img.shape[1])
                x1 = scale_position(xr[i+1][0], 1.5, img.shape[0])
                y1 = scale_position(xr[i+1][2], 1.5, img.shape[1])

                img = cv2.line(img, (x0, y0), (x1, y1), (1, 1, 1), 2)

            img = lqr_loop.render(img)
            img = mpc_loop.render(img)

            cv2.imshow("visualisation", img)
            cv2.waitKey(1)

            print("energy = ", round(lqr_loop.energy, 2), round(mpc_loop.energy, 2))



        n+= 1

