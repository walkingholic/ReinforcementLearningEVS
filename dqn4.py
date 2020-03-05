"""DQN Class
DQN(NIPS-2013)
"Playing Atari with Deep Reinforcement Learning"
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
DQN(Nature-2015)
"Human-level control through deep reinforcement learning"
http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
"""
import numpy as np
import tensorflow as tf


class DQN:

    def __init__(self, session: tf.Session, input_size: int, output_size: int, name: str="main") -> None:
        """DQN Agent can
        1) Build network
        2) Predict Q_value given state
        3) Train parameters
        Args:
            session (tf.Session): Tensorflow session
            input_size (int): Input dimension
            output_size (int): Number of discrete actions
            name (str, optional): TF Graph will be built under this name scope
        """
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()


    def _build_network(self, h_size= 18, l_rate=0.001) -> None:
        """DQN Network architecture (simple MLP)
        Args:
            h_size (int, optional): Hidden layer dimension
            l_rate (float, optional): Learning rate
        """
        with tf.variable_scope(self.net_name):

            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")
            net = self._X

            W1 = tf.get_variable("W1", shape=[self.input_size, h_size], initializer=tf.contrib.layers.xavier_initializer())
            layer1 = tf.nn.tanh(tf.matmul(net, W1))

            W2 = tf.get_variable("W2", shape=[h_size, h_size], initializer=tf.contrib.layers.xavier_initializer())
            layer2 = tf.nn.tanh(tf.matmul(layer1, W2))

            W3 = tf.get_variable("W3", shape=[h_size, h_size], initializer=tf.contrib.layers.xavier_initializer())
            layer3 = tf.nn.tanh(tf.matmul(layer2, W3))

            W4 = tf.get_variable("W4", shape=[h_size, self.output_size], initializer=tf.contrib.layers.xavier_initializer())

            self._Qpred = tf.matmul(layer3, W4)

            self._Y = tf.placeholder(tf.float32, shape=[None, self.output_size])
            self._loss = tf.losses.mean_squared_error(self._Y, self._Qpred)

            optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
            self._train = optimizer.minimize(self._loss)

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Returns Q(s, a)
        Args:
            state (np.ndarray): State array, shape (n, input_dim)
        Returns:
            np.ndarray: Q value array, shape (n, output_dim)
        """
        x = np.reshape(state, [-1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    # def predict_test(self, state: np.ndarray) -> np.ndarray:
    #     """Returns Q(s, a)
    #     Args:
    #         state (np.ndarray): State array, shape (n, input_dim)
    #     Returns:
    #         np.ndarray: Q value array, shape (n, output_dim)
    #     """
    #     x = np.reshape(state, [-1, self.input_size])
    #     return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack: np.ndarray, y_stack: np.ndarray) -> list:
        """Performs updates on given X and y and returns a result
        Args:
            x_stack (np.ndarray): State array, shape (n, input_dim)
            y_stack (np.ndarray): Target Q array, shape (n, output_dim)
        Returns:
            list: First element is loss, second element is a result from train step
        """
        feed = {
            self._X: x_stack,
            self._Y: y_stack
        }
        return self.session.run([self._loss, self._train], feed)