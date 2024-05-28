import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import quantecon as qe
import numpy as np

# Определим матрицу вероятностей перехода
transition_matrix = np.array([
    [3, 0, 2, 0, 2, 0, 0, 3],
    [0, 5, 0, 0, 0, 0, 5, 0],
    [3, 0, 3, 1, 0, 0, 0, 3],
    [0, 1, 0, 2, 2, 3, 2, 0],
    [4, 0, 1, 0, 3, 0, 0, 2],
    [0, 0, 3, 0, 3, 3, 1, 0],
    [0, 0, 0, 3, 3, 0, 3, 1],
    [2, 0, 4, 0, 2, 0, 0, 2]
]) / 10  # Делим на 10 для получения вероятностей

def generate_markov_chain(transition_matrix, initial_state, n_steps):
    n_states = transition_matrix.shape[0]
    states = np.zeros(n_steps, dtype=int)
    states[0] = initial_state
    for i in range(1, n_steps):
        states[i] = np.random.choice(n_states, p=transition_matrix[states[i-1]-1]) + 1
    return states

# Начальные состояния
initial_states = list(range(1, transition_matrix.shape[0] + 1))

# Длины цепей
chain_lengths = [10, 50, 100, 1000]

# Генерация и вывод траекторий
for length in chain_lengths:
    print(f"Траектории для цепи длины {length}:")
    for initial_state in initial_states:
        trajectory = generate_markov_chain(transition_matrix, initial_state, length)
        print(f"Состояние {initial_state}: {trajectory}")
    print()


# изменим название матрицы, чтобы не путаться
P = np.array([
    [3, 0, 2, 0, 2, 0, 0, 3],
    [0, 5, 0, 0, 0, 0, 5, 0],
    [3, 0, 3, 1, 0, 0, 0, 3],
    [0, 1, 0, 2, 2, 3, 2, 0],
    [4, 0, 1, 0, 3, 0, 0, 2],
    [0, 0, 3, 0, 3, 3, 1, 0],
    [0, 0, 0, 3, 3, 0, 3, 1],
    [2, 0, 4, 0, 2, 0, 0, 2]
]) / 10

# Create the Markov chain object
mc = qe.MarkovChain(P)

# Compute the stationary distribution
ψ_star = mc.stationary_distributions[0]

# Length of the time series
ts_length = 1000

# Simulate the Markov chain
X = mc.simulate(ts_length)

# Plotting setup
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_ylim(-0.25, 0.25)
ax.axhline(0, linestyle='dashed', lw=2, color='black', alpha=0.4)

# Векторы финальных состояний
final_probabilities = {
    1: 0.2938,
    2: 0.4444,
    3: 0.2574,
    4: 0,
    5: 0.1936,
    6: 0,
    7: 0.5555,
    8: 0.2551
}

# Loop through each state
for x0 in range(8):
    # Calculate the fraction of time for each state
    p_hat = (X == x0).cumsum() / (1 + np.arange(ts_length))

    # Print the empirical and theoretical values
    print(f'Состояние {x0 + 1}: эмпирическая частота = {p_hat[-1]}, стационарное распределение = {ψ_star[x0]},финальная вер-ть = {final_probabilities[x0+1]}')

    # Plot the difference between empirical and theoretical values
    ax.plot(p_hat - ψ_star[x0], label=f'$x = {x0 + 1} $')
    ax.set_xlabel('t')
    ax.set_ylabel(r'$\hat p_n(x) - \psi^* (x)$')

# Add legend and show plot
ax.legend()
plt.show()

