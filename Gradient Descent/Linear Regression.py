import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

class LinearReg:

    def __init__(self , epoch , LearningRate):
        self.b = -10
        self.m = -10
        self.epoch = epoch
        self.lr = LearningRate
        self.history = {'m':[] , 'b':[] , 'cost':[]}

    def fit(self , x , y):
        y = np.ravel(y)
        x = np.ravel(x)
        n = len(y)
        for i in range(self.epoch):
            slope_b = -2*(np.sum(y - x*self.m - self.b))/n
            slope_m = np.sum(-2*(y - x*self.m - self.b)*x)/n
            self.b = self.b - self.lr*slope_b
            self.m = self.m - self.lr*slope_m
            self.history['m'].append(self.m)
            self.history['b'].append(self.b)
            self.history['cost'].append(np.mean((y - self.m*x - self.b)**2))

    def intercepft_ (self):
        print(self.b)

    def coeff_ (self):
        print(self.m)

    def predict(self , x):
        return self.m*np.ravel(x) + self.b 
    
os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
df = pd.read_csv("study.csv")

x_train , x_test , y_train , y_test = train_test_split(df[['x']] , df[['y']] , test_size = 0.2 , random_state = 2)

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scale = scaler.transform(x_train)
x_test_scale = scaler.transform(x_test)

lr2 = LinearReg(250 , 0.01)
lr2.fit(x_train_scale , y_train)
y_pred2 = lr2.predict(x_test_scale)

fig , ax = plt.subplots(2 , 2 , figsize = (26 , 19))
ax[0,0].scatter(x_train_scale , y_train)
animated_line, = ax[0,0].plot([] , [] , 'r-' , linewidth = 2 , label = 'Fit')
ax[0,0].set_xlabel('X')
ax[0,0].set_ylabel('Y')
ax[0,0].set_title("Fit over epochs")
ax[0,0].grid()

animated_line_cost, = ax[0,1].plot([] , [] , 'b-' , linewidth = 2 , label = 'Cost')
ax[0,1].set_xlabel('Epoch')
ax[0,1].set_ylabel('MSE Cost')
ax[0,1].grid()
ax[0,1].set_title("Cost over epochs")
ax[0,1].set_xlim(0, lr2.epoch)                                                          
ax[0,1].set_ylim(min(lr2.history['cost']) * 0.9,max(lr2.history['cost']) * 1.1)        
  
animated_line_slope, = ax[1,0].plot([] , [] , 'b-' , linewidth = 2 , label = 'Slope')
ax[1,0].set_xlabel('Epoch')
ax[1,0].set_ylabel('Slope m')
ax[1,0].set_title('Slope over epochs')
ax[1,0].set_xlim(0 , lr2.epoch)
ax[1,0].set_ylim(min(lr2.history['m'])*0.9 , max(lr2.history['m'])*1.1)
ax[1,0].grid()

animated_line_intercept, = ax[1,1].plot([] , [] , 'b-' , linewidth = 2 , label = 'Slope')
ax[1,1].set_xlabel('Epoch')
ax[1,1].set_ylabel('Intercept b')
ax[1,1].grid()
ax[1,1].set_title("Intercept over epochs")
ax[1,1].set_xlim(0 , lr2.epoch)
ax[1,1].set_ylim(min(lr2.history['b'])*0.9 , max(lr2.history['b'])*1.1)

def update(frame):
    m = lr2.history['m'][frame]
    b = lr2.history['b'][frame]
    y_line = m * x_train_scale + b
    animated_line.set_data(x_train_scale , y_line)

    x = np.arange(0 , frame+1)
    cost_y = lr2.history['cost'][ : frame+1]
    animated_line_cost.set_data(x , cost_y)

    slope_y = lr2.history['m'][ : frame+1]
    animated_line_slope.set_data(x , slope_y)

    intercept_y = lr2.history['b'][ : frame+1]
    animated_line_intercept.set_data(x , intercept_y)

    return animated_line, animated_line_cost, animated_line_slope, animated_line_intercept,

animation = FuncAnimation(fig = fig , func = update , frames = 250 , interval = 4)

animation.save('Gradient_Descent.gif')
plt.show()