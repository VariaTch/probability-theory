import numpy as np
import scipy.linalg

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

def compute_stationary_distribution(transition_matrix):
    n = transition_matrix.shape[0]
    q = transition_matrix.T - np.eye(n)
    ones = np.ones((n, 1))
    A = np.vstack((q, ones.T))
    b = np.zeros((n + 1,))
    b[-1] = 1
    stationary_distribution = np.linalg.lstsq(A, b, rcond=None)[0]
    return stationary_distribution

# Длина траектории
n_steps = 100000

# Начальное состояние
initial_state1 = 1

# Генерация траектории
trajectory = generate_markov_chain(transition_matrix, initial_state1, n_steps)

# Подсчет времени нахождения в каждом состоянии
unique, counts = np.unique(trajectory, return_counts=True)
state_counts = dict(zip(unique, counts))

# Вычисление процента времени нахождения в каждом состоянии
percent_times = {state: count / n_steps * 100 for state, count in state_counts.items()}

# Вычисление стационарного распределения
stationary_distribution = compute_stationary_distribution(transition_matrix) * 100

# Вывод результатов
print("Процент времени нахождения (эмпирический):")
for state in range(1, transition_matrix.shape[0] + 1):
    print(f"State {state}: {percent_times.get(state, 0):.2f}%")

print("\nStationary distribution (theoretical):")
for state, prob in enumerate(stationary_distribution, 1):
    print(f"State {state}: {prob:.2f}%")

# Сравнение результатов
print("\nComparison (empirical vs theoretical):")
for state in range(1, transition_matrix.shape[0] + 1):
    empirical = percent_times.get(state, 0)
    theoretical = stationary_distribution[state - 1]
    print(f"State {state}: Empirical = {empirical:.2f}%, Theoretical = {theoretical:.2f}%")
