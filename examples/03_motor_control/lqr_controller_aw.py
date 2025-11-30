import LibsControl
import numpy
import matplotlib.pyplot as plt

#parameters from identification

#sampling frequency, 2kHz
dt       = 1.0/2000.0

u_max       =  1.0          # torque max value
k           =  205.443      # motor constant 
rad_var     =  29           # (rad/s)^2 variance (encoder noise)
tau         =  7.957        # motor time constant, milliseconds


rpm_max = u_max*k*60.0/(2.0*numpy.pi)

print("rpm_max = ", rpm_max)


#continuous time dynamics
mat_a = numpy.zeros((1, 1))
mat_b = numpy.zeros((1, 1))
mat_c = numpy.eye(1)
 
tau = tau*0.001
mat_a[0][0] = -1.0/tau    
mat_b[0][0] = k*(1.0/tau)


print("continuous model")
print("a = ", mat_a)
print("b = ", mat_b)
print()


#create dynamical system
ds = LibsControl.DynamicalSystem(mat_a, mat_b, mat_c, dt)

a_disc, b_disc, c_disc = LibsControl.c2d(ds.a, ds.b, ds.c, dt)

print("discrete model")
print("a = ", a_disc)
print("b = ", b_disc)
#print()
#print("poles")
#print(numpy.linalg.eigvals(a_disc))
print("\n")


aw_enabled = [False,  False, True]
aw_values  = [10**10, u_max, u_max]

labels     = ["no antiwindup", "clip only", "full antiwindup"]


t_result_all    = []
u_result_all    = []
xr_result_all   = []
x_result_all    = []

for k in range(len(aw_values)):
    #create loss weighting matrices (diagonal)
    q = numpy.array([ [1.0] ] )
    r = numpy.array( [ [(10**6)] ]) 

    #solve LQR controller
    lqr = LibsControl.LQRIDiscrete(a_disc, b_disc, q, r, aw_values[k], aw_enabled[k])

    print("controller")
    print("k  = ", lqr.k)
    print("ki = ", lqr.ki)



    #process simulation

    n = mat_a.shape[0]
    m = mat_b.shape[1]

    n_max = 500

    #required output, 1900 rpm
    rpm_req = 1900
    xr = numpy.zeros((n, 1))
    # convert rpm to rad/s
    xr_req_value = (rpm_req/60.0)*(2.0*numpy.pi) 

    #initial integral action
    integral_action = numpy.zeros((m, 1))

    #result log
    t_result    = []
    u_result    = []
    xr_result   = []
    x_result    = []


    #initial motor state
    ds.reset()

    for n in range(n_max):

        # unit step as input
        if n > 100 and n < 1500:
            xr[0][0] = xr_req_value
        else:
            xr[0][0] = 0

        #plant current state
        x = ds.x

        #compute controller output
        u, integral_action = lqr.forward(xr, x, integral_action)
        
        #compute plant output
        x, y = ds.forward_state(u)
        
        # store for plot
        t_result.append(n*dt)
        u_result.append(u[:, 0].copy())
        xr_result.append(xr[:, 0].copy())
        x_result.append(x[:, 0].copy())

        
    t_result    = numpy.array(t_result)
    u_result    = numpy.array(u_result)
    xr_result   = numpy.array(xr_result)
    x_result    = numpy.array(x_result)

    t_result_all.append(t_result)
    u_result_all.append(u_result)
    xr_result_all.append(xr_result)
    x_result_all.append(x_result)


t_result_all    = numpy.array(t_result_all)
u_result_all    = numpy.array(u_result_all)
xr_result_all   = numpy.array(xr_result_all)
x_result_all    = numpy.array(x_result_all)
    

xr_result_all*= 60.0/(2.0*numpy.pi)
x_result_all*= 60.0/(2.0*numpy.pi)




plt.clf()

count = 3
fig, axs = plt.subplots(count, 1, figsize=(8, 2*count))


for i in range(x_result_all.shape[0]):
    axs[0].plot(t_result, u_result_all[i, :, 0], label=labels[i], alpha=0.75)
    
    axs[0].set_xlabel("time [s]")
    axs[0].set_ylabel("controller output")
    axs[0].legend(loc='right')
    axs[0].grid()


axs[1].plot(t_result, xr_result_all[0, :, 0], color="black", label="required", alpha=0.75, lw=2.5)

for i in range(x_result_all.shape[0]):
    axs[1].plot(t_result, x_result_all[i, :, 0], label=labels[i], alpha=0.75)
    
    axs[1].set_xlabel("time [s]")
    axs[1].set_ylabel("velocity [rpm]")
    axs[1].legend(loc='right')
    axs[1].grid()

min_step = 150
max_step = 500

axs[2].plot(t_result[min_step:max_step], xr_result_all[0, min_step:max_step, 0], color="black", label="required", alpha=0.75, lw=2.5)

for i in range(x_result_all.shape[0]):
    axs[2].plot(t_result[min_step:max_step], x_result_all[i,  min_step:max_step, 0], label=labels[i], alpha=0.75)
    
    axs[2].set_xlabel("time [s]")
    axs[2].set_ylabel("velocity detail [rpm]")
    axs[2].legend(loc='right')
    axs[2].grid()



plt.tight_layout()
plt.savefig("plots/lqr_aw_result.png", dpi = 300)
