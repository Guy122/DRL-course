import gym
import numpy as np
import tensorflow.compat.v1 as tf
import collections

# optimized for Tf2
tf.disable_v2_behavior()
print("tf_ver:{}".format(tf.__version__))

env = gym.make('CartPole-v1')
np.random.seed(1)

tb_dir_str = "tb_dir"
ckpnt_sav_dir_str = "sav_dir"
ckpnt_sav_every_dir_str = "sav_every_dir"
ckpnt_load_dir_str = "shallow_sav_wts"
ckpnt_load_dir_str = "deep_sav_wts"
is_big = False
load_ckpnt = 1

do_train = False
LEARNING_RATE = 0.007
if not do_train:
    LEARNING_RATE = 0
NUM_OF_EPISODES = 5000
if do_train:
    NUM_OF_STEPS = 10000
else:
    NUM_OF_STEPS = 500
REPLAY_LIMIT = 20000
epsilon_decay = 0.999
min_epsilon = 0.01
if not do_train:
    min_epsilon = 0
SAMPLE_SIZE = 32
GAMMA = 0.95
C = 200
update_models_batch_size = SAMPLE_SIZE
# update_models_batch_size = 1
epsilon_wait_ts = 500


do_baseline = 1
do_actor_critic = 0


gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(gpus)

class PolicyNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        layer_width = 12
        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
            self.W1 = tf.get_variable("W1", [self.state_size, layer_width], initializer=tf2_initializer)
            self.b1 = tf.get_variable("b1", [layer_width], initializer=tf2_initializer)
            self.W2 = tf.get_variable("W2", [layer_width, self.action_size], initializer=tf2_initializer)
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf2_initializer)

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

class PolicyNetwork_bl:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        layer_width = 12
        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.delta = tf.placeholder(tf.float32, name="delta")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
            self.W1 = tf.get_variable("W1", [self.state_size, layer_width], initializer=tf2_initializer)
            self.b1 = tf.get_variable("b1", [layer_width], initializer=tf2_initializer)
            self.W2 = tf.get_variable("W2", [layer_width, self.action_size], initializer=tf2_initializer)
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf2_initializer)

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
class PolicyNetwork_ac:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        layer_width = 12
        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.delta = tf.placeholder(tf.float32, name="delta")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")
            self.I = tf.placeholder(tf.float32, name="I")

            tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
            self.W1 = tf.get_variable("W1", [self.state_size, layer_width], initializer=tf2_initializer)
            self.b1 = tf.get_variable("b1", [layer_width], initializer=tf2_initializer)
            self.W2 = tf.get_variable("W2", [layer_width, self.action_size], initializer=tf2_initializer)
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf2_initializer)

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.I * self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

class ValueNetwork_bl:
    def __init__(self, state_size, value_size, learning_rate, name='value_network'):
        self.state_size = state_size
        self.value_size = value_size
        self.learning_rate = learning_rate
        layer_width = 12
        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            #self.output = tf.placeholder(tf.int32, [self.value_size], name="value")
            #self.value = tf.placeholder(tf.int32, [self.value_size], name="value")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")
            #self.output = tf.placeholder(tf.float32, name="total_rewards")

            tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
            self.W1 = tf.get_variable("W1", [self.state_size, layer_width], initializer=tf2_initializer)
            self.b1 = tf.get_variable("b1", [layer_width], initializer=tf2_initializer)
            self.W2 = tf.get_variable("W2", [layer_width, self.value_size], initializer=tf2_initializer)
            self.b2 = tf.get_variable("b2", [self.value_size], initializer=tf2_initializer)

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)


            # Loss with mse
            #output = tf.make_ndarray(tf.convert_to_tensor(self.output, dtype=tf.float32))
            #output = tf.convert_to_tensor(self.output, dtype=tf.float32)
            #output = tf.make_ndarray(tf.Tensor(self.output.op.get_attr('value'))
            #self.loss = tf.keras.metrics.mean_squared_error(tf.reduce_mean(self.R_t, self.output))
            #self.neg_log_prob = tf.compat.v1.metrics.mean_squared_error(predictions=self.output, labels=self.R_t)
            #self.loss = tf.losses.mean_squared_error(predictions=self.output*(self.R_t - tf.stop_gradient(self.output)), labels=0)
            #self.delta = tf.stop_gradient(self.R_t - self.output)
            self.delta = self.R_t - self.output
            self.loss = tf.losses.mean_squared_error(predictions=self.delta, labels=0)
            #self.loss = self.output*(self.R_t - tf.stop_gradient(self.output))
            self.loss = tf.reduce_mean(self.loss)
            #self.loss = tf.reduce_mean((self.R_t-self.output)*self.output)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


class ValueNetwork:
    def __init__(self, state_size, learning_rate, name='state_value_network'):
        self.state_size = state_size
        self.learning_rate = learning_rate

        with tf.variable_scope("dadada"):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.next_output = tf.placeholder(tf.float32, name="next_output")
            self.A_t = tf.placeholder(tf.float32, name="discounted_advantage")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")
            self.I_factor = tf.placeholder(tf.float32, name="I_factor")
            self.gamma = tf.placeholder(tf.float32, name="I_factor")

            tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
            self.W1 = tf.get_variable("W1", [self.state_size, 64], initializer=tf2_initializer)
            self.b1 = tf.get_variable("b1", [64], initializer=tf2_initializer)
            self.W2 = tf.get_variable("W2", [64, 16], initializer=tf2_initializer)
            self.b2 = tf.get_variable("b2", [16], initializer=tf2_initializer)
            self.W3 = tf.get_variable("W3", [16, 1], initializer=tf2_initializer)
            self.b3 = tf.get_variable("b3", [1], initializer=tf2_initializer)

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            self.A2 = tf.nn.relu(self.Z2)
            self.output = tf.add(tf.matmul(self.A2, self.W3), self.b3)

            # Mean squared error loss
            self.mse = tf.losses.mean_squared_error(predictions=self.output, labels=self.R_t)
            #self.mse = tf.losses.mean_squared_error(predictions=self.output, labels=self.R_t+self.gamma*tf.stop_gradient(self.next_output))
            self.loss = tf.reduce_mean(self.mse * self.I_factor)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


def run():
    assert not (do_baseline and do_actor_critic),"do_baseline and do_actor_critic"
    writer = tf.summary.FileWriter(tb_dir_str)
    nof_steps_var = tf.Variable(0, dtype=tf.float32)  # variable that holds accuracy
    nof_steps_var_summ = tf.summary.scalar('Accuracy', nof_steps_var)
    sess_steps = tf.Session()
    loss_var = tf.Variable(0, dtype=tf.float32)  # variable that holds accuracy
    loss_var_summ = tf.summary.scalar('loss', loss_var)

    # Define hyperparameters
    state_size = 4
    value_size = 1
    action_size = env.action_space.n

    max_episodes = 5000
    max_steps = 501
    discount_factor = 0.99
    learning_rate1 = 0.0005
    learning_rate2 = 0.004

    render = False

    # Initialize the policy network
    tf.reset_default_graph()
    #ops.reset_default_graph()
    if do_baseline:
        policy = PolicyNetwork_bl(state_size, action_size, learning_rate1)
        value = ValueNetwork(state_size, value_size, learning_rate2)
    elif  do_actor_critic:
        policy = PolicyNetwork_ac(state_size, action_size, learning_rate1)
        value = ValueNetwork(state_size, learning_rate2)

    else:
        policy = PolicyNetwork(state_size, action_size, learning_rate1)
    # Start training the agent with REINFORCE algorithm
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        solved = False
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        episode_rewards = np.zeros(max_episodes)
        average_rewards = 0.0

        for episode in range(max_episodes):
            state = env.reset()
            state = state.reshape([1, state_size])
            episode_transitions = []
            I = 1

            for step in range(max_steps):
                actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
                action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.reshape([1, state_size])

                if render:
                    env.render()

                action_one_hot = np.zeros(action_size)
                action_one_hot[action] = 1
                episode_transitions.append(Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))
                episode_rewards[episode] += reward
                if do_actor_critic:
                    if 1:
                        feed_dict = {value.state: state}
                        value_current_state = sess.run(value.output, feed_dict)

                        # Calculate state-value output for next state
                        feed_dict = {value.state: next_state}
                        value_next_state = sess.run(value.output, feed_dict)

                        # calculate advantage
                        if done:
                            target = reward
                        else:
                            target = reward + discount_factor * value_next_state

                        advantage = target - value_current_state

                        # Update the value network weights
                        feed_dict = {value.state: state, value.A_t: advantage,
                                     value.R_t: target, value.I_factor: I,
                                     value.next_output: value_next_state, value.gamma: discount_factor}
                        _, loss_state = sess.run([value.optimizer, value.loss], feed_dict)
                    else:
                        feed_dict_v = {value.state: episode_transitions[-1].state, value.next_state: episode_transitions[-1].next_state, value.R_t: reward, value.gamma: discount_factor, value.I: I, value.done: episode_transitions[-1].done}
                        delta, delta_I = sess.run([value.delta, value.delta_I], feed_dict_v)
                        feed_dict = {policy.state: episode_transitions[-1].state, policy.action: episode_transitions[-1].action, policy.R_t: delta, policy.delta: delta, policy.I: I}
                        # feed_dict = {policy.state: episode_transitions[-1].state, policy.action: episode_transitions[-1].action, policy.delta: delta}
                        _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)
                        advantage = delta

                    feed_dict = {policy.state: episode_transitions[-1].state, policy.action: episode_transitions[-1].action, policy.R_t: advantage, policy.delta: advantage, policy.I:I}
                    #feed_dict = {policy.state: episode_transitions[-1].state, policy.action: episode_transitions[-1].action, policy.delta: delta}
                    _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)

                if done:
                    if episode > 98:
                        # Check if solved
                        average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                    print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode], round(average_rewards, 2)))
                    if average_rewards > 475:
                        print(' Solved at episode: ' + str(episode))
                        solved = True
                    break
                I = I*discount_factor
                state = next_state
            if solved:
                break

            if not do_actor_critic:
                # Compute Rt for each time-step t and update the network's weights
                for t, transition in enumerate(episode_transitions):
                    total_discounted_return = sum(discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:]))  # Rt
                    if not do_baseline:
                        feed_dict = {policy.state: transition.state, policy.R_t: total_discounted_return, policy.action: transition.action}
                        _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)
                    else:
                        feed_dict_v = {value.state: transition.state, value.R_t: total_discounted_return}#, value.R_t: total_discounted_return}
                        output = sess.run(value.output, feed_dict_v)
                        delta  = total_discounted_return - output

                        feed_dict = {policy.state: transition.state, policy.action: transition.action, policy.R_t: delta, policy.delta: delta}
                        _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)
                        #writer.add_summary(sess_steps.run(loss_var_summ), episode)  # add summary
                        #tf.summary.scalar(name="loss", data=float(loss), step=episode)
                        #writer.flush()

            sess_steps.run(nof_steps_var.assign(len(episode_transitions)))  # update accuracy variable
            writer.add_summary(sess_steps.run(nof_steps_var_summ), episode)  # add summary
            writer.flush()  # make sure everything is written to disk
if __name__ == '__main__':
    #writer = tf.summary.create_file_writer(logdir=tb_dir_str)
    #with writer.as_default():
    #with tf.device("cpu:0"):
        run()
