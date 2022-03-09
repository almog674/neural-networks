from cProfile import label
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

# Get 200 samples of sin waves
series = np.sin(0.1*np.arange(500)) + np.random.randn(500) * 0.1

# Build out dataset
T = 10
X = []
Y = []

for t in range(len(series) - T):
    x = series[t:t+T]
    X.append(x)
    y = series[t + T]
    Y.append(y)

X = np.array(X).reshape(-1, T)
Y = np.array(Y)
N = len(X)
print(f'X.shape: {X.shape}, Y.shape: {Y.shape}')


i = Input(shape=(T,))
x = Dense(1)(i)
model = Model(i, x)

model.compile(loss='mse', optimizer=Adam(lr=0.1))


r = model.fit(
    X[:-N//2], Y[:-N//2], epochs=80, validation_data=(X[-N//2:], Y[-N//2:])
)


validation_target = Y[-N//2:]
validation_predictions = []

last_x = X[-N//2]

while len(validation_predictions) < len(validation_target):
    p = model.predict(last_x.reshape(1, -1))[0, 0]
    i += 1

    validation_predictions.append(p)

    last_x = np.roll(last_x, -1)  # shift every element 1 spot to the left
    last_x[-1] = p


plt.plot(validation_target, label='target')
plt.plot(validation_predictions, label='predictions')
plt.legend()
plt.show()
