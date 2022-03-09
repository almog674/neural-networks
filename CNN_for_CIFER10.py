import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model


# Load & prepare the data
data = tf.keras.datasets.cifar10
(X_train, y_train), (X_test, y_test) = data.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()
print(X_train.shape)
#  X_train, X_test = X_train / 255.0, X_test / 255.0

K = len(set(y_train))


# Create the model
i = Input(shape=X_train[0].shape)
x = Conv2D(32, (3, 3), 2, activation='relu')(i)
x = Conv2D(64, (3, 3), 2, activation='relu')(x)
x = Conv2D(128, (3, 3), 2, activation='relu')(x)

x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)

model = Model(i, x)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15)

# Show the results
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# Get heatmap
y_predicted = model.predict(X_test)

y_flatten_predicted = [np.argmax(i) for i in y_predicted]

cm = tf.math.confusion_matrix(labels=y_test, predictions=y_flatten_predicted)
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d')
plt.show()
