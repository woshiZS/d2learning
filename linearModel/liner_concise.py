import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
from tensorflow._api.v2 import data
from tensorflow.python.eager.backprop_util import IsTrainable

true_w = tf.constant([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train = True):
    """Construct a tensorflow data iterator"""
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays) # Create a dataset from a python list or some array like objects
    if is_train:
        dataset = dataset.shuffle(buffer_size = 1000)
    dataset = dataset.batch(batch_size)
    return dataset

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# no need to specify input shape
# and initialization happens when codes execute.

initializer = tf.initializers.RandomNormal(stddev=0.01)
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1, kernel_initializer=initializer))

# loss = tf.keras.losses.MeanSquaredError()
loss = tf.keras.losses.Huber()
trainer = tf.keras.optimizers.SGD(learning_rate=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with tf.GradientTape() as tape:
            l = loss(net(X, training = True), y)
        grads = tape.gradient(l, net.trainable_variables)
        trainer.apply_gradients(zip(grads, net.trainable_variables))
    l = loss(labels, net(features))
    print(f'epoch {epoch + 1}, loss {l}')

w = net.get_weights()[0]
b = net.get_weights()[1]
print(f'error in estimating w: {tf.reshape(w, true_w.shape) - true_w}\nerror in estimating b: {b - true_b}')
