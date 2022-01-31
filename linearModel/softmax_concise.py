from re import L
import tensorflow as tf
import matplotlib.pyplot as plt
from d2l import tensorflow as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net = tf.keras.models.Sequential()
net.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
weight_initializer = tf.keras.initializers.RandomNormal(0.0, 0.01)
net.add(tf.keras.layers.Dense(10, kernel_initializer=weight_initializer))

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# trainer就相当于与优化函数了，封装得是真的方便
trainer = tf.keras.optimizers.SGD(learning_rate=0.1)

num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
plt.show()