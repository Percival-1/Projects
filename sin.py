import sympy as sm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation

x = np.linspace(0 , 40 , 501)

y = np.sin(x)

fig , ax = plt.subplots()

ax.set_xlim([0 , 41])
ax.set_ylim([-2 , 2])

animated_plot, = ax.plot([] , [])


def update(frame):
    animated_plot.set_data(x[:frame] , y[:frame])
    return animated_plot,

animation = FuncAnimation(
                fig = fig,
                func = update,
                frames = len(x),
                interval = 25,
                repeat = False
                )

plt.show()