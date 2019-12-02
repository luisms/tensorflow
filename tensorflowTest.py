import tensorflow as tf
import numpy as np
import datetime
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = np.reshape(x_train, (60000,28,28,1))
x_test = np.reshape(x_test, (10000,28,28,1))
print(np.shape(x_train))
print(np.shape(x_test))


print("modelo")
model = tf.keras.models.Sequential([
  
  tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=(28,28,1)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print (log_dir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

print("Train")
#model.fit(x_train, y_train, epochs=5,callbacks=[tensorboard_callback])
model.fit(x_train, y_train, epochs=5)
print("Evaluate")
model.evaluate(x_test,  y_test, verbose=2)