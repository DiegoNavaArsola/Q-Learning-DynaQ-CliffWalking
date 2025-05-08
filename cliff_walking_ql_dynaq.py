import gymnasium as gym # https://gymnasium.farama.org/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class QLearningAgent:
    """
    Clase para entrenear al agente con los algoritmos Q-Learning y Dyna-Q
    """
    def __init__(self,
                 env_name,
                 is_slippery=False,
                 gamma=0.99,
                 alpha=0.8,
                 epsilon_init=1,
                 epsilon_final=0.01,
                 epsilon_decay=0.001,
                 planning_steps=50,
                 dyna=False):

        self.is_slippery = is_slippery
        self.env_name = env_name

        # https://gymnasium.farama.org/
        self.env = gym.make(env_name, is_slippery=self.is_slippery)

        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.q_table = np.zeros((self.n_states, self.n_actions))

        self.alpha = alpha  # tasa de aprendizaje fija
        self.visit_counts = np.zeros((self.n_states, self.n_actions))  # Contador por par (s, a)

        self.gamma = gamma  # factor de descuento
        self.epsilon_max = epsilon_init # valor incial de epsilon
        self.decay_rate = epsilon_decay # parámetro que indica qué tan rápido decrece epsilon enc ada episodio
        self.epsilon_min = epsilon_final # valor final de epsilon

        # Modelo: diccionario de transiciones (s, a) -> (s', r)
        self.dyna = dyna
        self.model = {}
        self.planning_steps = planning_steps

    def epsilon_decay(self, episode):
        """
        Calcula el valor de epsilon para cada episodio según una función de decrecimiento exponencial
        """
        return self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-self.decay_rate * episode)

    def epsilon_greedy_policy(self, state, epsilon):
        """
        Determina una acción, dada una politica epsilon-greedy
        """
        # Si un valor aleatorio es menor a epsilon:
        if np.random.rand() < epsilon:

            # Ejecutamos una acción aleatoria
            return self.env.action_space.sample()

        # De lo contrario escogemos la mejor acción en ese estado
        return np.argmax(self.q_table[state])

    def train(self, n_episodes=1000):

        if self.dyna:
            print("\nIniciando entrenamiento con Dyna-Q...\n")
        else:
            print("\nIniciando entrenamiento con Q-Learning...\n")

        all_rewards = []

        for episode in range(n_episodes):

            # Reinicia el ambiente
            state = self.env.reset()[0]

            # Reiniciamos el valor de recompensa por episodio
            reward_per_episode = 0

            # Calcula la epsilon para el episodio actual
            epsilon = self.epsilon_decay(episode)

            while True:

                # Obtenemos la acción para el estado actual dada una politica epsilon-greedy
                # A <- epsilon-greedy(S,Q)
                action = self.epsilon_greedy_policy(state, epsilon)

                # Tomamos la acción A: Observamos R (Recompensa inmediata), S´(Siguiente estado).
                # R(s,a)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # Calculamos la tasa de aprendizaje para ese estado acción: alpha = 1 / (1 + N(s,a))
                self.visit_counts[state, action] += 1
                alpha = 1.0 / (1.0 + self.visit_counts[state, action])

                # Actualizamos la tabla Q: Q(S,A) <- Q(S,A)+ alpha[R + gamma * max_a(Q(S',A)) - Q(S,A)]
                self.q_table[state, action] += alpha * (
                            reward + self.gamma * np.max(self.q_table[next_state, :]) - self.q_table[state, action])

                # Para Dyna-Q (fase de planeación):
                if self.dyna:

                    # Obtenemos el estado siguiente y el valor de la recompensa desde modelo: Model(S,A) <- R,S'
                    self.model[(state, action)] = (next_state, reward)

                    for _ in range(self.planning_steps):
                        # Obtenemos un conjunto estado-acción ya observado de forma aleatoria
                        s, a = list(self.model.keys())[np.random.randint(len(self.model))]

                        # Obtenemos el estado siguiente y el valor de la recompensa desde modelo: R,S' <- Model(S,A)
                        s_next, r = self.model[(s, a)]

                        # Actualizamos la tabla Q.
                        self.q_table[s, a] += alpha * (
                                r + self.gamma * np.max(self.q_table[s_next, :]) - self.q_table[s, a])

                # El estado siguiente se convierte en el estado actual
                state = next_state

                # Actualizamos el valor de la recompensa acumulada en el episodio
                reward_per_episode += reward

                # Si el episodio termina o se trunca, salimos del loop y reiniciamos
                if terminated or truncated:
                    break

            if episode % 10 == 0:
                print(f"Epsiodio {episode}")

            all_rewards.append(reward_per_episode)

            avg_rewards_over_epsiodes = np.mean(all_rewards[-n_episodes:])

        return all_rewards, avg_rewards_over_epsiodes

    def get_qtable(self):
        return self.q_table

    def test(self, test_episodes=10, render_on=True):

        if render_on:
            render_mode = "human"
        else:
            render_mode = None

        env_test = gym.make(self.env_name,
                       is_slippery=self.is_slippery,
                       render_mode=render_mode)

        total_reward = []

        for episode in range(test_episodes):

            print(f"Test. Episodio: {episode + 1}")

            reward_episode = []

            state = env_test.reset()[0]

            # Genera un render del ambiente
            env_test.render()

            episode_over = False

            while not episode_over:
                # Seleccionamos la mejor acción dada la tabla Q
                action = np.argmax(self.q_table[state, :])

                # Observamos
                next_state, reward, terminated, truncated, info = env_test.step(action)
                state = next_state

                reward_episode.append(reward)

                episode_over = terminated or truncated

            total_reward.append(np.mean(reward_episode))

        env_test.close()

        return total_reward

def plot_rewards(qlearning_res, dynaq_res, windows_size=25):
    # Gráfica
    plt.plot(np.convolve(qlearning_res, np.ones(windows_size) / windows_size, mode='valid'), label="Q-Learning", color="blue")
    plt.plot(np.convolve(dynaq_res, np.ones(windows_size) / windows_size, mode='valid'), label="Dyna-Q", color="red")
    plt.legend()
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa promedio")
    plt.title("Q-learning y Dyna-Q para Cliff walking")
    plt.grid(True)
    plt.show()

def save_results(data, file_name):
    # Crear un DataFrame con un array
    df = pd.DataFrame(enumerate(data), columns=['experiment', 'mean_reward'])
    # Guardar en CSV
    df.to_csv(file_name+'.csv', index=False)
    print(f"Resultados gradados en: {file_name+".csv"}")


if __name__ == "__main__":

    n_episodes = 100
    planning_steps = 25
    env_name = "CliffWalking-v0"

    """
    Q-Learning
    """
    qlearning_agent = QLearningAgent(env_name,
                                     is_slippery=False,
                                     dyna=False)

    rewards_qlearning, _ = qlearning_agent.train(n_episodes)

    test_qlearning = qlearning_agent.test(3,render_on=True)

    print(f"Resultados en test: {test_qlearning}")

    save_results(test_qlearning, "qlearning")


    """
    Dyna-Q
    """
    dynaq_agent = QLearningAgent(env_name,
                                 is_slippery=False,
                                 planning_steps=planning_steps,
                                 dyna=True)

    rewards_dynaq, _ = dynaq_agent.train(n_episodes)

    test_dynaq = dynaq_agent.test(3,render_on=True)

    print(f"Resultados en test: {test_dynaq}")

    save_results(test_dynaq, "dynaq")


    # Graficar entrenamientos
    plot_rewards(rewards_qlearning, rewards_dynaq)
