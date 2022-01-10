 !apt-get update
 !apt-get install -y xvfb python-opengl ffmpeg
 !pip install pyglet==1.3.2
 !pip install gym pyvirtualdisplay
 from collections import deque
 import numpy as np
 import tensorflow as tf
 import random
 import math
 import time
 import os
 import io
 import glob
 import gym
 import base64
 from gym.wrappers import Monitor
 from IPython.display import HTML
 from IPython import display as ipythondisplay
 from pyvirtualdisplay import Display
 display = Display(visible=0, size=(1400, 900))
 display.start()
 
 env = gym.make('CartPole-v0')
 num_features = env.observation_space.shape[0]
 num_actions = env.action_space.n
 
 #El objetivo de la red neuronal mediante aprendizaje reforzado, es que aprenda a predecir la equacion de "Bellman", la cual consta de de 2 Q.
 #Q que representa el estado actual y Q' que representa el siguiente estado. 
 #Para lograr dicho cometido, nuestra red neuronal esta compuesta por dos redes neuronales de aprendizaje supervisado.
 #Una red neuronal Principal la cual se encarga de enseñarse a predecir el estado actual Q.
 #Y una red Objetivo la cual servira para predecir el siguiente estado "Q'", pero esta red neuronal no aprendera,
 #La red neuronal Objetivo recibira el conocimiento aprendido por la red neuronal principal, y con este conecimiento aprendera a predecir el siguiente estado "Q'".
 
 #Para poder entrenar la Red Neuronal principal y que esta aprenda a predecir el estado actual "Q". Se necesita de un dataset de datos de Q.
 #Por lo que es necesario crear un Buffer de experencias donde se guardara toda la informacion de los estados "Q".
 #Gracias a este Buffer se puede crear una coleccion de estados "Q" la cual servira para el entrenamiento la la red neuronal Principal. 
# 
 #Como obtenemos los estados Q para poder ingresarlos en nuestro buffer de experencias?
 #Muy facil se crea un algoritmo que ejecuta una politica de ejecucion de acciones denominada la politica e-greedy.
 #Esta politica es uno de los algoritmos de tipo voraz que existen para ejecutar acciones.
 #Es una estrategia de busqueda por la cual se sigue una heuristica consistente en elegir la opción optima en cada paso local con la esperanza de llegar a una solución general óptima. 
 #Esta política irá ejecutando acciones las cuales devolverán información del estado que se alcanza al ejecutar cierta accion, la recompensa per ejecutar cierta accion etc.
 #Toda esta información servirá para nutrir a nuestro buffer de información necesaria para entrenar la red neuronal principal.
 #En conclusión al ejecutar dicha política y almacenar su información en buffer estamos creando un dataset respecto al estado actual "Q" que servirá para entrenar a la red neuronal principal.
 
 #Nuestra politica e-greedy minimiza el factor epsilon que indica el grado de exploración o explotación. Donde la exploración representa las acciones aleatorias y la explotación las mejores acciones según la política.
 
 #Cada 2000 pasos de entrenamiento volcaremos el aprendizaje de la red neuronal principal en la objetivo.
 #Con la finalidad de mejorar las predicciones de la red neuronal principal en sus siguientes pasos de entrenamiento y así reducir recursos a la hora de continuar con el entrenamiento de la red principal y mejorar la precisión en las predicciones de la red neuronal principal.
 
 #Comenzaremos creando los DNN principal (main_nn) y objetivo (target_nn). 
 #La red de destino sera una copia de la principal, pero con su propia copia de
 #los pesos psinapticios.
 #Tambien necesitaremos un optimizador y una funcion de perdida.
 
 class DQN(tf.keras.Model):
   def __init__(self):
     super(DQN, self).__init__()
     self.dense1 = tf.keras.layers.Dense(32, activation="relu")#Capa numero 1 conta de 32 neuronas y funcion de activacion Reli.
     self.dense2 = tf.keras.layers.Dense(32, activation="relu")#Capa numero 2 conta de 32 neuronas y funcion de activacion Reli.
     self.dense3 = tf.keras.layers.Dense(num_actions, dtype=tf.float32) #Capa numero 3 conta de 32 neuronas y no tiene una funcion de activation.
 
   def call(self, x):
    # """Forward pass."""
     x = self.dense1(x)#Se ejecuta el Algoritmo de Fordward Pass enla primera capa y el resultado de los valoes de los pesos psinapticos es guardado en X.
     x = self.dense2(x)#Se ejecuta el Algoritmo de Fordward Pass enla primera capa y el resultado de los valoes de los pesos psinapticos es guardado en X.
     return self.dense3(x)#Se ejecuta el Algoritmo de Fordward Pass enla primera capa y el resultado de la preddion es devuelto para ser usado en el BacWardPass.
 
 main_nn = DQN()#Red Neuronal de aprendizaje supervisado Principal.
 target_nn = DQN()#Red Neuronal de aprendizaje supervisado Secundaria.
 
 optimizer = tf.keras.optimizers.Adam(1e-4) #Optimizador.
 #Funcion de perdida.
 mse = tf.keras.losses.MeanSquaredError() #Obtenemos el Error Cuadratico Medio.
 
 
 #A continuaciñon, crearemos el búfer de reproducción de experiencias para agregar la
 #Experencia al bufer y probarlo mas tarde para el entrenamiento.
 class ReplayBuffer(object):
   def __init__(self, size):#Se inicia el Buffer
     self.buffer = deque(maxlen=size)
   def add(self, state, action, reward, next_state, done):#Se añaden datos al buffer.
     self.buffer.append((state, action, reward, next_state, done))
   def __len__(self):#Tamaño del buffer.
     return len(self.buffer)
 
   def sample(self, num_samples):#Obtenemos una muestra de tamaño "num_samples" de datos coleccion de datos en el buffer.
     states, actions, rewards, next_states, dones = [], [], [], [], []
     #Obtenemos una matriz aleatoria de tamaño "num_samples" con indices aleatorios del buffer.
     idx = np.random.choice(len(self.buffer), num_samples)#Las posiciones de los datos del buffer que se van a sustraer.
     for i in idx:#Recorremos los indices guardados en la matriz idx.
       elem = self.buffer[i]#Obtenemos los elemntos pertenecientes al indice dentro del buffer.
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
   #Se crea un tensor de 1 dimension con un numero aleatorio desde 0 a 1.
   result = tf.random.uniform((1,))#Obtenemos un numero aleatorio desde 0 a 1.
   if result < epsilon: #El ratio de aprendizaje es mayor al numero aleatorio?
     #Si.
     return env.action_space.sample()#Devolvemos un estado aleatorio dentro del juego,
                                     #Porque la red neuronal aun no ha experimentado lo suficiente con el juego.
   else:
    #No.
     return tf.argmax(main_nn(state)[0]).numpy()#la red neuronal ha aprendido lo suficiente del juego como para tomar sus propias decisiones.
 
 
 #Para calcular el estado actual "Q" necesitamos saber el valor del estado siguiente "Q'" por lo que para obtener el valor de "Q'" se realiza una predicción en la red neuronal Objetivo.
 #Para realizar la predicción de los siguientes estados "Q'" se necesita tener un dataset de los "Q'". El cual Obtenemos gracias nuestro bufger de experencias.
 @tf.function
 def train_step(states, actions, rewards, next_states, dones):
   next_qs = target_nn(next_states)#Aplicamos FordwardPass en la red neuronal Objetivo, usando como datos de entrenamiento el dataset de "Q'".
   max_next_qs = tf.reduce_max(next_qs, axis=-1)#Obtenemos el siguiente estado "Q'"  con valor maximo.
   #Q = recompensa + Factor_Descuento * maxQ' * (1. - False==0)
   target = reward + discount * max_next_qs * (1. - dones)#Calculamos Q y cambiamos el valor de done a True.
   with tf.GradientTape() as tape:
     qs = main_nn(states)
     #se crea una matriz de acciones
     actions_masks = tf.one_hot(actions, num_actions)
     #Se suma .....
     #...
     masked_qs = tf.reduce_sum(actions_masks * qs, axis =- 1)
     #Obtenemos el error a partir de el calculo del Error Cuadratico Medio.
     loss = mse(target, masked_qs)#Obtenemos el error Cuadratico medio del estado atual "Q" comparandolo con la matriz de acciones.
   grads = tape.gradient(loss, main_nn.trainable_variables)#Se calcula el Gradiente.
   optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))
   return loss
 
 
 #También definiremos los hiperparámetros necesarios y entrenaremos la red
 #neuronal. Reproduciremos un episodio usando la política ε-greedy, almacenaremos
 #los datos en el búfer de reproducción de experiencias y entrenaremos la red
 #principal después de cada paso. Una vez cada 2000 pasos, copiaremos los pesos
 #de la red principal a la red de destino. También disminuiremos el valor de
 #épsilon (ε) para comenzar con una exploración alta y disminuiremos la exploración
 #con el tiempo. Veremos cómo el algoritmo empieza a aprender después de cada episodio.
 
 num_episodes = 1000
 epsilon = 1.0 #Ratio de aprendizaje.
 batch_size = 32#Tamaño minimo del buffer.
 discount = 0.99 #Factor Descuento.
 buffer = ReplayBuffer(100000)
 cur_frame = 0#Contador.
 
 last_100_ep_rewards = []
 
 for episode in range(num_episodes+1): #1000Episodios
   #Obtenemos la observacin del entorno de ejecucion.
   state = env.reset()#Se prepara la variable estado para poder recibir estaos, tambien se reinicia.
   ep_reward, done = 0, False
   #While solo deja de ejecutarse cuando ¿Toda la matriz tiene Done == False?
   while not done:#Mientras done == False
     #Redimension de la matriz state añadiendo un 1 al principio de la matriz.
     #Se readapta la matriz state para poder ser usada en el metodo "select_epsilon_greedy_action".
     state_in = tf.expand_dims(state, axis=0)#Obtenemos el estado actual.
     #Introducimos el estado actual para obtener la accion que se desea realizar.
     action = select_epsilon_greedy_action(state_in, epsilon)#Obtenemos la accion al ejecutar el metodo que selcciona acciones aleatorias o las mejores.
     #Introducimos la accion en el juego y recibimos el siguiente estado, la recompensa del estado actual y done para ...
     #Recibimos parametros del juego "Entorno."
     next_state, reward, done, info = env.step(action)#Obtenemos la informacion del juego despues de haber realizado cierta accion dentro de el."
     ep_reward += reward#Recojemos la recompensa.
     #Guardamos en el buffer la experencia obtenida.
     buffer.add(state, action, reward, next_state, done) #Añadios los parametros del estado recibido por el juego, en nuestro buffer.
     state = next_state#Actualizamos el estado al siguiente estado.
     cur_frame += 1
     #Copiamos los pesos psinapticos de la red principal "main_nn" hacia la objetivo "target_nn".
     if cur_frame % 2000 == 0:
       #Tensorflow nos ofrece set_weights y get_weights.
       target_nn.set_weights(main_nn.get_weights())#Copiamos los pesos psinapticos de la red principal a la red objetivo.
 
       #Entrenando Red neuronal.
     if len(buffer) >= batch_size:#Se entrena la red neuronal si el buffer tiene la cantidad de datos minima necesaria.
       states, actions, rewards, next_states, dones = buffer.sample(batch_size)#Se obtiene los parametros necesarios del buffer para entrenar la red principal.
       #Se entrena la red neuronal principal.
       loss = train_step(states, actions, rewards, next_states, dones)#Obtenemos el error cuadratico medio.
 
   if episode < 950:
     #A partir del episodio 950 se reduce el ratio de aprendizaje en 0.0001,
     #Para que asi el descenso del gradiente cada vez de pasos mas pequeños y
     #Aproximarnos mejor a un error local.
     epsilon -= 0.001
 
   if len(last_100_ep_rewards) == 100: # Si hay 100 recompensas obtenidas.
     last_100_ep_rewards = last_100_ep_rewards[1:]
   last_100_ep_rewards.append(ep_reward)
 
   if episode % 50 == 0:
     print(f'Episode {episode}/{num_episodes}. Epsilon: {epsilon:.3f}. '
           f'Reward in last 100 episodes: {np.mean(last_100_ep_rewards):.3f}')
 env.close()

def show_video():
  """Enables video recording of gym environment and shows it."""
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
  state = tf.expand_dims(state, axis=0)
  action = select_epsilon_greedy_action(state, epsilon=0.01)
  state, reward, done, info = env.step(action)
  ep_rew += reward
print('Return on this episode: {}'.format(ep_rew))
env.close()
show_video()
