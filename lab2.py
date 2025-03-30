import math
import random
import time
import matplotlib.pyplot as plt

# ===================== FUNKCJE POMOCNICZE =====================

def read_tsplib(filename):
    coords = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    reading_coords = False
    for line in lines:
        line = line.strip()
        if line.startswith("NODE_COORD_SECTION"):
            reading_coords = True
            continue
        if line.startswith("EOF") or "DISPLAY_DATA_SECTION" in line:
            break
        if reading_coords:
            parts = line.split()
            if len(parts) >= 3:
                x = float(parts[1])
                y = float(parts[2])
                coords.append((x, y))
    return coords

def compute_distance_matrix(coords):
    n = len(coords)
    dist_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                dist_matrix[i][j] = 0
            else:
                (x1, y1) = coords[i]
                (x2, y2) = coords[j]
                dist = math.dist((x1, y1), (x2, y2))
                dist_matrix[i][j] = round(dist)
    return dist_matrix

def cycle_length(distance_matrix, cycle):
    length = 0
    for i in range(len(cycle)):
        current_node = cycle[i]
        next_node = cycle[(i+1) % len(cycle)]
        length += distance_matrix[current_node][next_node]
    return length

def solution_cost(distance_matrix, cycle1, cycle2):
    return cycle_length(distance_matrix, cycle1) + cycle_length(distance_matrix, cycle2)

def evaluate_solution(distance_matrix, cycle1, cycle2):
    # Funkcja celu: im mniejszy koszt, tym większa wartość (bo negacja)
    return - solution_cost(distance_matrix, cycle1, cycle2)

def random_solution(distance_matrix):
    n = len(distance_matrix)
    nodes = list(range(n))
    random.shuffle(nodes)
    if n % 2 == 0:
        size1 = n // 2
        size2 = n // 2
    else:
        size1 = n // 2 + 1
        size2 = n // 2
    cycle1 = nodes[:size1]
    cycle2 = nodes[size1:size1+size2]
    random.shuffle(cycle1)
    random.shuffle(cycle2)
    return cycle1, cycle2

def weighted_regret_two_cycles(distance_matrix, wcost = 1, wregret = -1):
    n = len(distance_matrix)

    if n % 2 == 0:
        size1 = n // 2
        size2 = n // 2
    else:
        size1 = n // 2 + 1
        size2 = n // 2

    nodes = list(range(n))

    random.shuffle(nodes)

    cycle1 = [nodes[0], nodes[1]]
    cycle2 = [nodes[2], nodes[3]]

    used = set([nodes[0], nodes[1], nodes[2], nodes[3]])
    remaining = nodes[4:]

    def insertion_cost(distance_matrix, cycle, i, v):
        """
        Koszt wstawienia v pomiędzy cycle[i] i cycle[i+1].
        """
        ncyc = len(cycle)
        i_next = (i + 1) % ncyc
        return (distance_matrix[cycle[i]][v] +
                distance_matrix[v][cycle[i_next]] -
                distance_matrix[cycle[i]][cycle[i_next]])

    def find_best_two_insertions(distance_matrix, cycle, v, wcost, wregret):
        best_cost = float('inf')
        scd_best_cost = float('inf')
        best_idx = None
        scd_best_idx = None
        for i in range(len(cycle)):
            cost = insertion_cost(distance_matrix, cycle, i, v)
            if cost < scd_best_cost:
                if cost < best_cost:
                    scd_best_cost = best_cost
                    best_cost = cost
                    best_idx = i
                    scd_best_idx = best_idx
                else:
                    scd_best_cost = cost
                    scd_best_idx = i

        regret = scd_best_cost - best_cost if scd_best_cost < float('inf') else 0

        score = wcost * cost + wregret * regret

        return best_cost, best_idx, score

    best_v, best_cycle, best_idx = None, None, None
    max_score = float('-inf')
    while remaining:
        best_v, best_cycle, best_idx = None, None, None
        max_score = float('-inf')
        cost1, idx1, score1 = float('inf'), None, float('-inf')
        cost2, idx2, score2 = float('inf'), None, float('-inf')

        for v in remaining:
            if len(cycle1) < size1:
                cost1, idx1, score1 = find_best_two_insertions(distance_matrix, cycle1, v, wcost, wregret)

            if len(cycle2) < size2:
                cost2, idx2, score2 = find_best_two_insertions(distance_matrix, cycle2, v, wcost, wregret)

            if score1 > max_score or (score1 == max_score and cost1 < cost2):
                best_v, best_cycle, best_idx, max_score = v, cycle1, idx1, score1

            if score2 > max_score or (score2 == max_score and cost2 < cost1):
                best_v, best_cycle, best_idx, max_score = v, cycle2, idx2, score2

        if best_v is not None:
            best_cycle.insert(best_idx + 1, best_v)
            remaining.remove(best_v) 

    return cycle1, cycle2

# ===================== FUNKCJE DELTY I RUCHÓW =====================

def delta_exchange_vertices_between_cycles(distance_matrix, cycle1, cycle2, idx1, idx2, current_eval):
    old_v1 = cycle1[idx1]
    old_v2 = cycle2[idx2]
    c1_copy = cycle1[:]
    c2_copy = cycle2[:]
    c1_copy[idx1], c2_copy[idx2] = old_v2, old_v1
    new_eval = evaluate_solution(distance_matrix, c1_copy, c2_copy)
    delta = new_eval - current_eval
    return delta, new_eval

def apply_exchange_vertices_between_cycles(cycle1, cycle2, idx1, idx2):
    cycle1[idx1], cycle2[idx2] = cycle2[idx2], cycle1[idx1]

def delta_swap_vertices_in_cycle(distance_matrix, cycle1, cycle2, which_cycle, i, j, current_eval):
    c1_copy = cycle1[:]
    c2_copy = cycle2[:]
    if which_cycle == 1:
        c1_copy[i], c1_copy[j] = c1_copy[j], c1_copy[i]
    else:
        c2_copy[i], c2_copy[j] = c2_copy[j], c2_copy[i]
    new_eval = evaluate_solution(distance_matrix, c1_copy, c2_copy)
    delta = new_eval - current_eval
    return delta, new_eval

def apply_swap_vertices_in_cycle(cycle, i, j):
    cycle[i], cycle[j] = cycle[j], cycle[i]

def delta_2opt_in_cycle(distance_matrix, cycle1, cycle2, which_cycle, i, j, current_eval):
    c1_copy = cycle1[:]
    c2_copy = cycle2[:]
    if which_cycle == 1:
        c1_copy[i+1:j+1] = reversed(c1_copy[i+1:j+1])
    else:
        c2_copy[i+1:j+1] = reversed(c2_copy[i+1:j+1])
    new_eval = evaluate_solution(distance_matrix, c1_copy, c2_copy)
    delta = new_eval - current_eval
    return delta, new_eval

def apply_2opt_in_cycle(cycle, i, j):
    cycle[i+1:j+1] = reversed(cycle[i+1:j+1])

# ===================== LOKALNE PRZESZUKIWANIE (STROME) =====================

def local_search_steepest(distance_matrix, cycle1, cycle2, use_2opt=False):
    improved = True
    current_eval = evaluate_solution(distance_matrix, cycle1, cycle2)
    n1 = len(cycle1)
    n2 = len(cycle2)
    while improved:
        improved = False
        best_delta = 0.0
        best_move_type = None
        best_indices = None
        # Ruchy między cyklami
        for i in range(n1):
            for j in range(n2):
                delta, _ = delta_exchange_vertices_between_cycles(distance_matrix, cycle1, cycle2, i, j, current_eval)
                if delta > best_delta:
                    best_delta = delta
                    best_move_type = "exchange"
                    best_indices = (i, j)
        # Ruchy wewnątrz cyklu
        if not use_2opt:
            # swap w cycle1
            for i in range(n1):
                for j in range(i+1, n1):
                    delta, _ = delta_swap_vertices_in_cycle(distance_matrix, cycle1, cycle2, 1, i, j, current_eval)
                    if delta > best_delta:
                        best_delta = delta
                        best_move_type = "swap_in_cycle"
                        best_indices = (1, i, j)
            # swap w cycle2
            for i in range(n2):
                for j in range(i+1, n2):
                    delta, _ = delta_swap_vertices_in_cycle(distance_matrix, cycle1, cycle2, 2, i, j, current_eval)
                    if delta > best_delta:
                        best_delta = delta
                        best_move_type = "swap_in_cycle"
                        best_indices = (2, i, j)
        else:
            # 2-opt w cycle1
            for i in range(n1):
                for j in range(i+2, n1):
                    delta, _ = delta_2opt_in_cycle(distance_matrix, cycle1, cycle2, 1, i, j, current_eval)
                    if delta > best_delta:
                        best_delta = delta
                        best_move_type = "2opt_in_cycle"
                        best_indices = (1, i, j)
            # 2-opt w cycle2
            for i in range(n2):
                for j in range(i+2, n2):
                    delta, _ = delta_2opt_in_cycle(distance_matrix, cycle1, cycle2, 2, i, j, current_eval)
                    if delta > best_delta:
                        best_delta = delta
                        best_move_type = "2opt_in_cycle"
                        best_indices = (2, i, j)
        if best_delta > 0:
            if best_move_type == "exchange":
                i, j = best_indices
                apply_exchange_vertices_between_cycles(cycle1, cycle2, i, j)
            elif best_move_type == "swap_in_cycle":
                which_cycle, i, j = best_indices
                if which_cycle == 1:
                    apply_swap_vertices_in_cycle(cycle1, i, j)
                else:
                    apply_swap_vertices_in_cycle(cycle2, i, j)
            elif best_move_type == "2opt_in_cycle":
                which_cycle, i, j = best_indices
                if which_cycle == 1:
                    apply_2opt_in_cycle(cycle1, i, j)
                else:
                    apply_2opt_in_cycle(cycle2, i, j)
            current_eval += best_delta
            improved = True
    return cycle1, cycle2

# ===================== ALGORYTM LOSOWEGO BŁĄDZENIA (RANDOM WALK) =====================

def random_walk(distance_matrix, cycle1, cycle2, use_2opt=False, max_iter=100):
    """
    W każdej iteracji wybieramy losowo jeden z ruchów:
      - exchange: losowo wybieramy i z cycle1 i j z cycle2
      - ruch wewnątrz tras:
          * jeśli use_2opt==False: losowy swap w losowo wybranym cyklu,
          * jeśli use_2opt==True: losowy ruch 2-opt w losowo wybranym cyklu.
    Bez względu na delta, ruch jest wykonywany, a jeśli nowe rozwiązanie jest lepsze – zapamiętujemy je.
    """
    current_c1 = cycle1[:]
    current_c2 = cycle2[:]
    best_c1 = current_c1[:]
    best_c2 = current_c2[:]
    best_eval = evaluate_solution(distance_matrix, best_c1, best_c2)
    n1 = len(current_c1)
    n2 = len(current_c2)
    for _ in range(max_iter):
        move_type = random.choice(["exchange", "intra"])
        if move_type == "exchange":
            i = random.randrange(n1)
            j = random.randrange(n2)
            apply_exchange_vertices_between_cycles(current_c1, current_c2, i, j)
        else:
            if not use_2opt:
                which_cycle = random.choice([1,2])
                if which_cycle == 1 and n1 >= 2:
                    i, j = random.sample(range(n1), 2)
                    if i > j: i, j = j, i
                    apply_swap_vertices_in_cycle(current_c1, i, j)
                elif which_cycle == 2 and n2 >= 2:
                    i, j = random.sample(range(n2), 2)
                    if i > j: i, j = j, i
                    apply_swap_vertices_in_cycle(current_c2, i, j)
            else:
                which_cycle = random.choice([1,2])
                if which_cycle == 1 and n1 >= 3:
                    i = random.randrange(n1-2)
                    j = random.randrange(i+2, n1)
                    apply_2opt_in_cycle(current_c1, i, j)
                elif which_cycle == 2 and n2 >= 3:
                    i = random.randrange(n2-2)
                    j = random.randrange(i+2, n2)
                    apply_2opt_in_cycle(current_c2, i, j)
        current_eval = evaluate_solution(distance_matrix, current_c1, current_c2)
        if current_eval > best_eval:
            best_eval = current_eval
            best_c1 = current_c1[:]
            best_c2 = current_c2[:]
    return best_c1, best_c2

# ===================== FUNKCJA RYSUJĄCA CYKLE =====================

def drawCycles(coords, cycle1, cycle2, title=""):
    plt.figure()
    x1 = [coords[i][0] for i in cycle1] + [coords[cycle1[0]][0]]
    y1 = [coords[i][1] for i in cycle1] + [coords[cycle1[0]][1]]
    plt.plot(x1, y1, marker='o', label='Cykl 1')
    x2 = [coords[i][0] for i in cycle2] + [coords[cycle2[0]][0]]
    y2 = [coords[i][1] for i in cycle2] + [coords[cycle2[0]][1]]
    plt.plot(x2, y2, marker='o', label='Cykl 2')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

# ===================== EKSPERYMENT =====================

def run_experiment(distance_matrix, runs=100):
    # Klucze eksperymentów: (start_type, move_type)
    # Dodajemy też kombinacje dla random walk ("rw")
    experiments = {
        ("random", "swap"): {"f_vals": [], "times": [], "best": None},
        ("random", "2opt"): {"f_vals": [], "times": [], "best": None},
        ("heur", "swap"): {"f_vals": [], "times": [], "best": None},
        ("heur", "2opt"): {"f_vals": [], "times": [], "best": None},
        ("rw", "swap"): {"f_vals": [], "times": [], "best": None},
        ("rw", "2opt"): {"f_vals": [], "times": [], "best": None},
    }
    best_overall = {key: -float('inf') for key in experiments}
    # Ustalmy dla random walk stałą liczbę iteracji
    rw_iter = 1000
    for combo in experiments:
        start_type, move_type = combo
        for _ in range(runs):
            # Wybieramy rozwiązanie startowe:
            if start_type == "random" or start_type == "rw":
                c1, c2 = random_solution(distance_matrix)
            elif start_type == "heur":
                c1, c2 = weighted_regret_two_cycles(distance_matrix)
            start_time = time.time()
            if start_type == "rw":
                final_c1, final_c2 = random_walk(distance_matrix, c1[:], c2[:], use_2opt=(move_type=="2opt"), max_iter=rw_iter)
            else:
                if move_type == "swap":
                    final_c1, final_c2 = local_search_steepest(distance_matrix, c1[:], c2[:], use_2opt=False)
                else:
                    final_c1, final_c2 = local_search_steepest(distance_matrix, c1[:], c2[:], use_2opt=True)
            elapsed = time.time() - start_time
            f_val = evaluate_solution(distance_matrix, final_c1, final_c2)
            experiments[combo]["f_vals"].append(f_val)
            experiments[combo]["times"].append(elapsed)
            if f_val > best_overall[combo]:
                best_overall[combo] = f_val
                experiments[combo]["best"] = (final_c1, final_c2, f_val)
    return experiments

def print_results_table(experiments):
    print("Wyniki funkcji celu:")
    print("{:<15} {:<10} {:<15} {:<15} {:<15}".format("Start", "Move", "Min f(x)", "Mean f(x)", "Max f(x)"))
    for combo, data in experiments.items():
        start_type, move_type = combo
        f_vals = data["f_vals"]
        print("{:<15} {:<10} {:<15.2f} {:<15.2f} {:<15.2f}".format(
            start_type, move_type, min(f_vals), sum(f_vals)/len(f_vals), max(f_vals)))
    print("\nCzasy obliczeń:")
    print("{:<15} {:<10} {:<15} {:<15} {:<15}".format("Start", "Move", "Min time (s)", "Mean time (s)", "Max time (s)"))
    for combo, data in experiments.items():
        start_type, move_type = combo
        times = data["times"]
        print("{:<15} {:<10} {:<15.4f} {:<15.4f} {:<15.4f}".format(
            start_type, move_type, min(times), sum(times)/len(times), max(times)))

# ===================== MAIN =====================

def main():
    try:
        filename = "kroB200.tsp"
        coords = read_tsplib(filename)
        print("Wczytano instancję z", filename)
    except Exception as e:
        print("Nie udało się wczytać instancji TSPLIB, używamy przykładowych danych.")
        coords = [(0,0), (10,0), (10,10), (0,10), (5,5)]
    distance_matrix = compute_distance_matrix(coords)
    runs = 10  # dla testów
    experiments = run_experiment(distance_matrix, runs)
    print_results_table(experiments)
    for combo, data in experiments.items():
        start_type, move_type = combo
        best_sol = data["best"]
        if best_sol is not None:
            best_c1, best_c2, f_val = best_sol
            title = f"{start_type}, {move_type}: f(x) = {f_val:.2f}, cost = {-f_val:.2f}"
            drawCycles(coords, best_c1, best_c2, title=title)

if __name__ == "__main__":
    main()
