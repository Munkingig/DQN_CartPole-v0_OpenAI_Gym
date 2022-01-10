
# %%capture
# !apt-get update
# !apt-get install -y xvfb python-opengl ffmpeg
# !pip install pyglet==1.3.2
# !pip install gym pyvirtualdisplay
# from collections import deque
# import numpy as np
# import tensorflow as tf
# import random
# import math
# import time
# import os
# import io
# import glob
# import gym
# import base64
# from gym.wrappers import Monitor
# from IPython.display import HTML
# from IPython import display as ipythondisplay
# from pyvirtualdisplay import Display
# display = Display(visible=0, size=(1400, 900))
# display.start()
# 
# env = gym.make('CartPole-v0')
# num_features = env.observation_space.shape[0]
# num_actions = env.action_space.n
# 
# 
# #Para implementar el algoritmo DQN, comenzaremos creando los DNN principal (main_nn) 
# #Y objetivo (target_nn). La red de destino sera una copia de la principal, pero con 
# #Su propia copia de los pesos. Tambien necesitaremos un optimizador y una funcion de perdida.
# 
# class DQN(tf.keras.Model):
#   """Dense neural network class."""
#   def __init__(self):
#     super(DQN, self).__init__()
#     self.dense1 = tf.keras.layers.Dense(32, activation="relu")
#     self.dense2 = tf.keras.layers.Dense(32, activation="relu")
#     self.dense3 = tf.keras.layers.Dense(num_actions, dtype=tf.float32) # No activation
# 
#   def call(self, x):
#     """Forward pass."""
#     x = self.dense1(x)
#     x = self.dense2(x)
#     return self.dense3(x)
# 
# main_nn = DQN()
# target_nn = DQN()
# 
# optimizer = tf.keras.optimizers.Adam(1e-4) #Optimizador
# mse = tf.keras.losses.MeanSquaredError() #Funcion de perdida.


#A continuaciñon, crearemos el búfer de reproducción de experiencias para agregar la
#Experencia al bufer y probarlo mas tarde para el entrenamiento.

class ReplayBuffer(object):
  """Experince replay buffer that samples uniformly"""
  def __init__(self, size):
    self.buffer = deque(maxlen=size)
  def add(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))
  def __len__(self):
    return len(self.buffer)

  def sample(self, num_samples):
    states, actions, rewards, next_states, dones = [], [], [], [], []
    idx = np.random.choice(len(self.buffer), num_samples)

    for i in idx:
      elem = self.buffer[i]
      state, action, reward, next_state, done = elem
      states.append(np.array(state, copy=False))
      actions.append(np.array(action, copy=False))
      rewards.append(reward)
      next_states.append(np.array(next_state, copy=False))
      dones.append(done)
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards, dtype=np.float32)
    next_states = np.array(next_states)
    dones = np.array(dones, dtype=np.float32)
    return states, actions, rewards, next_states, dones

#Tambien escribiremos una funcion auxiliar para ejecutar la politica e-greedy y
#para entrenar la red principal usando los datos almacenados en el bufer.

def select_epsilon_greedy_action(state, epsilon):
  result = tf.random.uniform((1,))
  if result < epsilon:
    return env.action_space.sample()
  else:
    return tf.argmax(main_nn(state)[0]).numpy()

@tf.function
def train_step(states, actions, rewards, next_states, dones):
  next_qs = target_nn(next_states)
  max_next_qs = tf.reduce_max(next_qs, axis=-1)
  #Q = recompensa + Factor_Descuento * maxQ' * (1. - False==0)
  target = reward + discount * max_next_qs * (1. - dones)
  with tf.GradientTape() as tape:
    qs = main_nn(states)
    #se crea una matriz de acciones
    #....
    actions_masks = tf.one_hot(actions, num_actions)
    #Se suma .....
    #...
    masked_qs = tf.reduce_sum(actions_masks * qs, axis =- 1)
    #Obtenemos el error a partir de el calculo del Error Cuadratico Medio.
    loss = mse(target, masked_qs)
  grads = tape.gradient(loss, main_nn.trainable_variables)
  optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))
  return loss

#También definiremos los hiperparámetros necesarios y entrenaremos la red
#neuronal. Reproduciremos un episodio usando la política ε-greedy, almacenaremos
#los datos en el búfer de reproducción de experiencias y entrenaremos la red
#principal después de cada paso. Una vez cada 2000 pasos, copiaremos los pesos
#de la red principal a la red de destino. También disminuiremos el valor de
#épsilon (ε) para comenzar con una exploración alta y disminuiremos la exploración
#con el tiempo. Veremos cómo el algoritmo empieza a aprender después de cada episodio

num_episodes = 1000
epsilon = 1.0 #Ratio de aprendizaje
batch_size = 32#
discount = 0.99 #Factor Descuento.
buffer = ReplayBuffer(100000)
cur_frame = 0#Contador 
#env = gym.make('CartPole-v0')

last_100_ep_rewards = []
for episode in range(num_episodes+1):
  #Obtenemos la observacin del entorno de ejecucion.
  state = env.reset()
  ep_reward, done = 0, False
  #While solo deja de ejecutarse cuando ¿Toda la matriz tiene Done == False?
  while not done:
    #Redimension de la matriz state añadiendo un 1 al principio de la matriz.
    state_in = tf.expand_dims(state, axis=0)
    action = select_epsilon_greedy_action(state_in, epsilon)
    next_state, reward, done, info = env.step(action)
    ep_reward += reward
    #Guardamos en el buffer la experencia obtenida.
    buffer.add(state, action, reward, next_state, done)
    state = next_state
    cur_frame += 1
    #Copiamos los pesos psinapticos de la red principal "main_nn" hacia la objetivo "target_nn"
    if cur_frame % 2000 == 0:
      #Tensorflow nos ofrece set_weights y get_weights.
      target_nn.set_weights(main_nn.get_weights())

      #Entrenando Red neuronal.
    if len(buffer) >= batch_size:
      states, actions, rewards, next_states, dones = buffer.sample(batch_size)
      #Obtenemos el error cuadratico medio.
      loss = train_step(states, actions, rewards, next_states, dones)

  if episode < 950:
    #A partir del episodio 950 se reduce el ratio de aprendizaje en 0.0001,
    #Para que asi el descenso del gradiente cada vez de pasos mas pequeños y
    #Aproximarnos mejor a un error local.
    epsilon -= 0.001

  if len(last_100_ep_rewards) == 100:
    last_100_ep_rewards = last_100_ep_rewards[1:]
  last_100_ep_rewards.append(ep_reward)

  if episode % 50 == 0:
    print(f'Episode {episode}/{num_episodes}. Epsilon: {epsilon:.3f}. '
          f'Reward in last 100 episodes: {np.mean(last_100_ep_rewards):.3f}')
env.close()

def show_video():
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
  else: 
    print("Video not found")

def wrap_env(env):
  env = Monitor(env, './video', force=True)
  return env

env = wrap_env(gym.make('CartPole-v0'))
state = env.reset()
done = False
ep_rew = 0
while not done:
  env.render()
  state = state.astype(np.float32)
  #state = torch.from_numpy(np.expand_dims(state, axis=0)).to(device)
  state = tf.expand_dims(state, axis=0)
  action = select_epsilon_greedy_action(state, epsilon=0.01)
  state, reward, done, info = env.step(action)
  ep_rew += reward
print('Return on this episode: {}'.format(ep_rew))
env.close()
show_video()

