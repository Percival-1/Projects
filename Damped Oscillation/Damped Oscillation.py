import sympy as sm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation

# constants
k, c, m = sm.symbols('k, c, m' , positive = True , real = True)

# Variables
t = sm.symbols('t' , positive = True , real = True)

# functions x(t) and F_ext(t)
x = sm.Function('x' , complex = True)(t)
F_ext = sm.Function('F_{ext}' , real = True)(t)

# Derivatives
dx_dt = sm.diff(x , t , 1) # First
ddx_ddt = sm.diff(x , t , 2) # Second

# Equation for damped oscillation
equation = ddx_ddt + (c/m)*dx_dt + (k/m)*x - F_ext

# specifying the initial conditions
init_cond = {x.subs(t , 0) : 0,
             dx_dt.subs(t , 0) : 2}

solution = sm.dsolve(equation , x , ics = init_cond)

# f_ext = 0
new_F_ext = 0 # N
m_val = 2 # kg
k_val = 4 # N.m
c_val = 0.3 # SI unit

# solution after putting constant values
sol_sub = solution.subs([(F_ext , new_F_ext) , (m , m_val) , (k , k_val) , (c , c_val)]).simplify()

# making expression to vary with t
sol = sol_sub.rhs
solutions = sm.lambdify(t , sol)

# Display the results
t_var = np.linspace(0 , 90 , 501)

x_var = np.real(solutions(t_var))

fig , ax = plt.subplots(1 , 2 ,figsize = (10 , 6))

ax[0].set_xlim([-2 , 2])
ax[0].set_ylim([-4 , 4])
ax[0].grid(True)
ax[0].set_xlabel('Displacement x(t)')
ax[0].set_ylabel('Time t in sec')

ax[1].set_xlim([0 , 90])
ax[1].set_ylim([-2 , 2])
ax[1].grid(True)
ax[1].set_xlabel('Time t in sec')
ax[1].set_ylabel('Displacement x(t)')

animate_spring, = ax[0].plot([] , [] , color = 'blue')

animate_mass, = ax[0].plot([] , [] , 'o' , markersize = 20 , color = 'red')

animated_plot, = ax[1].plot([] , [])


def update(frame):
    animated_plot.set_data(t_var[:frame] , x_var[:frame])
    animate_mass.set_data([x_var[frame]] , [0])
    animate_spring.set_data([-2 , x_var[frame]] , [0 , 0])
    animate_spring.set_linewidth(int(abs(x_var[frame] - 2)*2))
    return animated_plot, animate_mass, animate_spring

animation = FuncAnimation(
                fig = fig,
                func = update,
                frames = len(t_var),
                interval = 25,
                repeat = False
                )
animation.save('d.gif')

plt.show()