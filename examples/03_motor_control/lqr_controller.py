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


#compute step response
t_result = []
u_result = []
x_result = []

for n in range(500):

    # unit step as input
    if n > 100 and n < 400:
        u = numpy.ones((1, 1))
    else:
        u = numpy.zeros((1, 1))

    x, _ = ds.forward_state(u)

    t_result.append(n*dt)
    u_result.append(u[:, 0])
    x_result.append(x[:, 0])

t_result = numpy.array(t_result)
x_result = numpy.array(x_result)

#plot result
LibsControl.plot_response(t_result, u_result,  x_result*60.0/(2.0*numpy.pi), "plots/step_response.png", ["control u"], ["rpm"])



a_disc, b_disc, c_disc = LibsControl.c2d(ds.a, ds.b, ds.c, dt)

print("discrete model")
print("a = ", a_disc)
print("b = ", b_disc)
#print()
#print("poles")
#print(numpy.linalg.eigvals(a_disc))
print("\n")


#create loss weighting matrices (diagonal)
q = numpy.array([ [1.0] ] )
r = numpy.array( [ [4*(10**7)] ]) 

#solve LQR controller
lqr = LibsControl.LQRIDiscrete(a_disc, b_disc, q, r, u_max)

print("controller")
print("k  = ", lqr.k)
print("ki = ", lqr.ki)






#process simulation

n = mat_a.shape[0]
m = mat_b.shape[1]

n_max = 2000

#required output, 1000rpm
rpm_req = 1000
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
    if n > 200 and n < 1500:
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

    
    
t_result = numpy.array(t_result)
xr_result = numpy.array(xr_result)
x_result = numpy.array(x_result)
u_result = numpy.array(u_result)

#convert rps to rpm
xr_result[:, 0]*= 60.0/(2.0*numpy.pi)
x_result[:, 0]*= 60.0/(2.0*numpy.pi)

#plot results
LibsControl.plot_cl_response(t_result, u_result, xr_result, x_result, "plots/lqr_result.png",  ["input u"], ["velocity [rpm]"])

