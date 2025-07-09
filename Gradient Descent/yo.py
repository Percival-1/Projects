import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class LinearReg:

    def __init__(self , epoch , LearningRate):
        self.b = -10
        self.m = -10
        self.epoch = epoch
        self.lr = LearningRate
        self.history = {'m':[] , 'b':[]}

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

    def intercepft_ (self):
        print(self.b)

    def coeff_ (self):
        print(self.m)

    def predict(self , x):
        return self.m*np.ravel(x) + self.b 
    
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# 1. Load and split
df = pd.read_csv("study.csv")
X_train, X_test, y_train, y_test = train_test_split(df[['x']], df['y'], test_size=0.2, random_state=2)

# 2. Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 3. Train with a higher learning rate
lr2 = LinearReg(epoch=5000, LearningRate=0.01)
lr2.fit(X_train_scaled, y_train.values.reshape(-1,1))

# 4. Recover original‐scale parameters
#    Because you trained on (x − μ)/σ, the true slope is m/σ and the intercept is b − m*μ/σ
m_scaled = lr2.m / scaler.scale_[0]
b_scaled = lr2.b - lr2.m * scaler.mean_[0] / scaler.scale_[0]

print(f"Slope: {m_scaled:.4f}, Intercept: {b_scaled:.4f}")
