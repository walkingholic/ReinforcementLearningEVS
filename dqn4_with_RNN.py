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


    def _build_network(self, h_size= 64, l_rate=0.001) -> None:
        """DQN Network architecture (simple MLP)
        Args:
            h_size (int, optional): Hidden layer dimension
            l_rate (float, optional): Learning rate
        """
        with tf.variable_scope(self.net_name):
            # input place holders

            seq_length = 24
            data_dim = 1
            hidden_dim = 128
            output_dim = 3
            learning_rate = 0.01
            self._X = tf.placeholder(tf.float32, [None, 26])
            net = self._X
            net = tf.reshape(net, [-1, 26, 1])

            # build a LSTM network
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
            outputs, _states = tf.nn.dynamic_rnn(cell, net[:, 2:], dtype=tf.float32)
            ev_state = tf.reshape(net[:, 0:2], [-1, 2])
            input_concat = tf.concat([ev_state, outputs[:, -1]], 1)

            # self._Qpred = tf.contrib.layers.fully_connected(input_concat, output_dim, activation_fn=None)

            input_concat = tf.layers.dense(input_concat, h_size, activation=tf.nn.relu)
            input_concat = tf.layers.dense(input_concat, self.output_size)
            self._Qpred = input_concat




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
        # print('predict')
        x = np.reshape(state, [-1, self.input_size])
        # print('*************-----predict x ')
        # print(np.shape(x))
        return self.session.run(self._Qpred, feed_dict={self._X: x})


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