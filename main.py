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

# Классы состояний
E0 = [4]
E1 = [6]
E2 = [2, 7]
E3 = [1, 3, 5, 8]

# Векторы финальных состояний
x0 = [0]
x1 = [0]
x2 = [4/9, 5/9]
x3 = [129/439, 113/439, 85/439, 112/439]

# Длина траектории
n_steps = 100000

# Начальное состояние
initial_state = 1

# Генерация траектории
trajectory = generate_markov_chain(transition_matrix, initial_state, n_steps)

# Подсчет времени нахождения в каждом состоянии
unique, counts = np.unique(trajectory, return_counts=True)
state_counts = dict(zip(unique, counts))

# Вычисление процента времени нахождения в каждом состоянии
percent_times = {state: count / n_steps * 100 for state, count in state_counts.items()}

# Сравнение с векторами финальных состояний
def compare_with_final_vectors(percent_times, classes, final_vectors):
    comparison = {}
    for class_states, final_vector in zip(classes, final_vectors):
        class_states_tuple = tuple(class_states)  # Преобразуем список в кортеж
        total_empirical = sum(percent_times.get(state, 0) for state in class_states)
        comparison[class_states_tuple] = {
            'empirical': [percent_times.get(state, 0) / total_empirical if total_empirical else 0 for state in class_states],
            'theoretical': final_vector
        }
    return comparison

classes = [E0, E1, E2, E3]
final_vectors = [x0, x1, x2, x3]

comparison = compare_with_final_vectors(percent_times, classes, final_vectors)

# Вывод результатов
print("Percentage of time in each state (empirical):")
for state in range(1, transition_matrix.shape[0] + 1):
    print(f"State {state}: {percent_times.get(state, 0):.2f}%")

print("\nComparison with final vectors:")
for class_states, result in comparison.items():
    print(f"Class {class_states}:")
    for state, (empirical, theoretical) in zip(class_states, zip(result['empirical'], result['theoretical'])):
        print(f"State {state}: Empirical = {empirical * 100:.2f}%, Theoretical = {theoretical * 100:.2f}%")
