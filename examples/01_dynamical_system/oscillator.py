import numpy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import LibsControl




if __name__ == "__main__":
    k  = 1
    dt = 0.01

    A = numpy.array([
        [0, k],
        [-k, 0]
    ])

    B = numpy.array([
        [0.0],
        [0.0]
    ])


    ds_a = LibsControl.DynamicalSystem(A, B, None, dt)
    ds_b = LibsControl.DynamicalSystem(A, B, None, dt)

    xa = numpy.zeros((2, 1))
    xa[0, 0] = 1

    xb = numpy.zeros((2, 1))
    xb[0, 0] = 1

    u = numpy.zeros((1, 1))

    xa_result = []
    xb_result = []
    for n in range(10000):
        xa, _ = ds_a.forward(xa, u, False)
        xb, _ = ds_b.forward(xb, u, True)

        xa_result.append(xa[:, 0].copy())
        xb_result.append(xb[:, 0].copy())

    xa_result = numpy.array(xa_result)
    xb_result = numpy.array(xb_result)

        
    # Create a figure with a custom grid
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 2, width_ratios=[1, 1.2], height_ratios=[1, 1], figure=fig)

    # --- 1. Phase-space plot (large square on left) ---
    ax0 = fig.add_subplot(gs[:, 0])  # spans both rows (tall)
    ax0.plot(xa_result[:, 0], xa_result[:, 1], label="Euler solver", color="royalblue")
    ax0.plot(xb_result[:, 0], xb_result[:, 1], label="Runge-Kutta 4 solver", color="red")
    ax0.set_xlabel("x₀ (Position)")
    ax0.set_ylabel("x₁ (Velocity)")
    ax0.set_aspect("equal", adjustable="box")  # keeps it square
    ax0.legend()
    ax0.grid(True)
    ax0.set_title("Phase Space (x₀ vs x₁)")

    # --- 2. Position vs time (top-right) ---
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(xa_result[:, 0], color="royalblue")
    ax1.plot(xb_result[:, 0], color="red")
    ax1.set_ylabel("x₀ (Position)")
    #ax1.legend()
    ax1.grid(True)
    ax1.set_title("Position vs Time")

    # --- 3. Velocity vs time (bottom-right) ---
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(xa_result[:, 1], color="royalblue")
    ax2.plot(xb_result[:, 1], color="red")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("x₁ (Velocity)")
    #ax2.legend()
    ax2.grid(True)
    ax2.set_title("Velocity vs Time")

    plt.tight_layout()
    plt.show()