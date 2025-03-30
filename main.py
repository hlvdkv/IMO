import math
import random
import time

import matplotlib.pyplot as plt

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
                # parts[0] - indeks, parts[1] - x, parts[2] - y
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


def nearest_neighbor_two_cycles(distance_matrix):

    n = len(distance_matrix)
    
    if n % 2 == 0:
        size1 = n // 2
        size2 = n // 2
    else:
        size1 = n // 2 + 1
        size2 = n // 2
    
    # Losowe startowe wierzchołki dla cykli
    all_nodes = list(range(n))
    start1 = random.choice(all_nodes)
    all_nodes.remove(start1)
    start2 = random.choice(all_nodes)
    all_nodes.remove(start2)
    
    cycle1 = [start1]
    cycle2 = [start2]
    
    used = set([start1, start2]) 

    def find_closest(current_node, distance_matrix, used):
        dmin = float('inf')
        best_node = None
        for v in range(len(distance_matrix)):
            if v not in used:
                if distance_matrix[current_node][v] < dmin:
                    dmin = distance_matrix[current_node][v]
                    best_node = v
        return best_node

    while len(cycle1) < size1:
        current = cycle1[-1]
        next_node = find_closest(current, distance_matrix, used)
        cycle1.append(next_node)
        used.add(next_node)

    while len(cycle2) < size2:
        current = cycle2[-1]
        next_node = find_closest(current, distance_matrix, used)
        cycle2.append(next_node)
        used.add(next_node)

    return cycle1, cycle2


def greedy_cycle_two_cycles(distance_matrix):

    n = len(distance_matrix)
    
    if n % 2 == 0:
        size1 = n // 2
        size2 = n // 2
    else:
        size1 = n // 2 + 1
        size2 = n // 2
    
    nodes = list(range(n))
    
    # losowe wierzchołki
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
        i_next = (i+1) % ncyc
        return (distance_matrix[cycle[i]][v] +
                distance_matrix[v][cycle[i_next]] -
                distance_matrix[cycle[i]][cycle[i_next]])

    def find_best_insertion(distance_matrix, cycle, v):
        best_cost = float('inf')
        best_idx = None
        for i in range(len(cycle)):
            cost = insertion_cost(distance_matrix, cycle, i, v)
            if cost < best_cost:
                best_cost = cost
                best_idx = i
        return best_cost, best_idx
    
    for v in remaining:
        if len(cycle1) < size1 and len(cycle2) < size2:
            cost1, idx1 = find_best_insertion(distance_matrix, cycle1, v)
            cost2, idx2 = find_best_insertion(distance_matrix, cycle2, v)
            
            if cost1 < cost2:
                cycle1.insert(idx1+1, v)
            else:
                cycle2.insert(idx2+1, v)
        elif len(cycle1) < size1:
            _, idx = find_best_insertion(distance_matrix, cycle1, v)
            cycle1.insert(idx+1, v)
        elif len(cycle2) < size2:
            _, idx = find_best_insertion(distance_matrix, cycle2, v)
            cycle2.insert(idx+1, v)
        else:
            # Oba cykle pełne (teoretycznie nie powinno się zdarzyć przy poprawnym rozkładzie).
            pass

    return cycle1, cycle2



#let's go
def regret_two_cycles(distance_matrix):
    n = len(distance_matrix)

    if n % 2 == 0:
        size1 = n // 2
        size2 = n // 2
    else:
        size1 = n // 2 + 1
        size2 = n // 2

    nodes = list(range(n))

    # losowe wierzchołki
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

    def find_best_two_insertions(distance_matrix, cycle, v):
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

        return best_cost, best_idx, regret

    best_v, best_cycle, best_idx = None, None, None
    max_regret = float('-inf')
    while remaining:
        best_v, best_cycle, best_idx = None, None, None
        max_regret = float('-inf')
        cost1, idx1, regret1 = float('inf'), None, float('-inf')
        cost2, idx2, regret2 = float('inf'), None, float('-inf')

        for v in remaining:
            if len(cycle1) < size1:
                cost1, idx1, regret1 = find_best_two_insertions(distance_matrix, cycle1, v)

            if len(cycle2) < size2:
                cost2, idx2, regret2 = find_best_two_insertions(distance_matrix, cycle2, v)

            if regret1 > max_regret or (regret1 == max_regret and cost1 < cost2):
                best_v, best_cycle, best_idx, max_regret = v, cycle1, idx1, regret1

            if regret2 > max_regret or (regret2 == max_regret and cost2 < cost1):
                best_v, best_cycle, best_idx, max_regret = v, cycle2, idx2, regret2

        if best_v is not None:
            best_cycle.insert(best_idx + 1, best_v)
            remaining.remove(best_v)  # Ensure we remove the inserted node

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

    # losowe wierzchołki
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
            remaining.remove(best_v)  # Ensure we remove the inserted node

    return cycle1, cycle2


def drawCycles(coords, cycle1, cycle2, title=""):
    """Funkcja rysująca dwa cykle na jednym wykresie."""
    plt.figure()
    # Przygotowanie współrzędnych dla cyklu 1 (zamknięty cykl)
    x1 = [coords[i][0] for i in cycle1] + [coords[cycle1[0]][0]]
    y1 = [coords[i][1] for i in cycle1] + [coords[cycle1[0]][1]]
    plt.plot(x1, y1, marker='o', label='Cykl 1')
    
    # Przygotowanie współrzędnych dla cyklu 2 (zamknięty cykl)
    x2 = [coords[i][0] for i in cycle2] + [coords[cycle2[0]][0]]
    y2 = [coords[i][1] for i in cycle2] + [coords[cycle2[0]][1]]
    plt.plot(x2, y2, marker='o', label='Cykl 2')
    
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

def greedy_vertice_regret(distance_matrix, cycle1, cycle2):

    improved = True
    while improved:
        improved = False
        l1 = cycle_length(distance_matrix, cycle1)
        l2 = cycle_length(distance_matrix, cycle2)
        for i in range(len(cycle1)):
            for j in range(len(cycle2)):
                test_cycle1 = cycle1.copy()
                test_cycle2 = cycle2.copy()
                test_cycle1[i] = cycle2[j]
                test_cycle2[j] = cycle1[i]
                tl1 = cycle_length(distance_matrix, test_cycle1)
                tl2 = cycle_length(distance_matrix, test_cycle2)
                if tl2+tl1 < l1+l2:
                    cycle1 = test_cycle1
                    cycle2 = test_cycle2
                    improved = True
                    break
            if improved:

                break

    return cycle1, cycle2

def greedy_inner_regret(distance_matrix, cycle1, cycle2):
    improved = True
    while improved:
        improved = False
        l1 = cycle_length(distance_matrix, cycle1)
        l2 = cycle_length(distance_matrix, cycle2)

        indices1 = list(range(len(cycle1)))
        indices2 = list(range(len(cycle2)))

        random.shuffle(indices1)
        random.shuffle(indices2)

        for i in indices1:
            for j in indices1:
                if i >= j:
                    continue

                test_cycle1 = cycle1.copy()
                test_cycle1[i], test_cycle1[j] = test_cycle1[j], test_cycle1[i]

                tl1 = cycle_length(distance_matrix, test_cycle1)
                if tl1 < l1:
                    cycle1 = test_cycle1
                    improved = True
                    break
            if improved:
                break

        for i in indices2:
            for j in indices2:
                if i >= j:
                    continue

                test_cycle2 = cycle2.copy()
                test_cycle2[i], test_cycle2[j] = test_cycle2[j], test_cycle2[i]

                tl2 = cycle_length(distance_matrix, test_cycle2)
                if tl2 < l2:
                    cycle2 = test_cycle2
                    improved = True
                    break
            if improved:
                break

    return cycle1, cycle2

def random_cycles(distance_matrix):
    n = len(distance_matrix)
    cycle1, cycle2 = [], []

    if n % 2 == 0:
        size1 = n // 2
        size2 = n // 2
    else:
        size1 = n // 2 + 1
        size2 = n // 2

    nodes = list(range(n))

    # losowe wierzchołki
    random.shuffle(nodes)
    for i in range(size1):
        cycle1.append(nodes[i])

    for i in range(size2):
        cycle2.append(nodes[size1+i])

    return(cycle1, cycle2)




def lab1():
    # Dla instancji kroA200.tsp
    filename = "kroA200.tsp"
    coords = read_tsplib(filename)

    distance_matrix = compute_distance_matrix(coords)
    n = len(distance_matrix)

    nn = []
    nn_results = []
    gc = []
    gc_results = []
    r = []
    r_results = []
    wr = []
    wr_results = []

    for _ in range(100):
        # Heurystyka najbliższego sąsiada
        cycle1_nn, cycle2_nn = nearest_neighbor_two_cycles(distance_matrix)
        length1_nn = cycle_length(distance_matrix, cycle1_nn)
        length2_nn = cycle_length(distance_matrix, cycle2_nn)
        total_nn = length1_nn + length2_nn
        nn.append(total_nn)
        nn_results.append((total_nn, cycle1_nn, cycle2_nn))

        # Heurystyka rozbudowy cyklu
        cycle1_gc, cycle2_gc = greedy_cycle_two_cycles(distance_matrix)
        length1_gc = cycle_length(distance_matrix, cycle1_gc)
        length2_gc = cycle_length(distance_matrix, cycle2_gc)
        total_gc = length1_gc + length2_gc
        gc.append(total_gc)
        gc_results.append((total_gc, cycle1_gc, cycle2_gc))

        # Heurystyka rozbudowy cyklu + żal
        cycle1_r, cycle2_r = regret_two_cycles(distance_matrix)
        length1_r = cycle_length(distance_matrix, cycle1_r)
        length2_r = cycle_length(distance_matrix, cycle2_r)
        total_r = length1_r + length2_r
        r.append(total_r)
        r_results.append((total_r, cycle1_r, cycle2_r))

        # Heurystyka rozbudowy cyklu + żal + wagi
        cycle1_wr, cycle2_wr = weighted_regret_two_cycles(distance_matrix)
        length1_wr = cycle_length(distance_matrix, cycle1_wr)
        length2_wr = cycle_length(distance_matrix, cycle2_wr)
        total_wr = length1_wr + length2_wr
        wr.append(total_wr)
        wr_results.append((total_wr, cycle1_wr, cycle2_wr))

    print("KroA200:")
    print("Nearest Neighbor: min =", min(nn), "mean =", sum(nn) / len(nn), "max =", max(nn))
    print("Greedy Cycle: min =", min(gc), "mean =", sum(gc) / len(gc), "max =", max(gc))
    print("Regret: min =", min(r), "mean =", sum(r) / len(r), "max =", max(r))
    print("Weighted Regret: min =", min(wr), "mean =", sum(wr) / len(wr), "max =", max(wr))

    # Wizualizacja dla najlepszych (najkrótszych) dróg instancji kroA200:
    best_nn = min(nn_results, key=lambda x: x[0])
    # drawCycles(coords, best_nn[1], best_nn[2], title="kroA200 - Nearest Neighbor, droga = " + str(best_nn[0]))

    best_gc = min(gc_results, key=lambda x: x[0])
    drawCycles(coords, best_gc[1], best_gc[2], title="kroA200 - Greedy Cycle, droga = " + str(best_gc[0]))

    best_r = min(r_results, key=lambda x: x[0])
    drawCycles(coords, best_r[1], best_r[2], title="kroA200 - Regret, droga = " + str(best_r[0]))

    best_wr = min(wr_results, key=lambda x: x[0])
    drawCycles(coords, best_wr[1], best_wr[2], title="kroA200 - Weighted Regret, droga = " + str(best_wr[0]))

    # Dla instancji kroB200.tsp
    filename = "kroB200.tsp"
    coords = read_tsplib(filename)

    distance_matrix = compute_distance_matrix(coords)
    n = len(distance_matrix)

    nn = []
    nn_results = []
    gc = []
    gc_results = []
    r = []
    r_results = []
    wr = []
    wr_results = []

    for _ in range(100):
        # Heurystyka najbliższego sąsiada
        cycle1_nn, cycle2_nn = nearest_neighbor_two_cycles(distance_matrix)
        length1_nn = cycle_length(distance_matrix, cycle1_nn)
        length2_nn = cycle_length(distance_matrix, cycle2_nn)
        total_nn = length1_nn + length2_nn
        nn.append(total_nn)
        nn_results.append((total_nn, cycle1_nn, cycle2_nn))

        # Heurystyka rozbudowy cyklu
        cycle1_gc, cycle2_gc = greedy_cycle_two_cycles(distance_matrix)
        length1_gc = cycle_length(distance_matrix, cycle1_gc)
        length2_gc = cycle_length(distance_matrix, cycle2_gc)
        total_gc = length1_gc + length2_gc
        gc.append(total_gc)
        gc_results.append((total_gc, cycle1_gc, cycle2_gc))

        # Heurystyka rozbudowy cyklu + żal
        cycle1_r, cycle2_r = regret_two_cycles(distance_matrix)
        length1_r = cycle_length(distance_matrix, cycle1_r)
        length2_r = cycle_length(distance_matrix, cycle2_r)
        total_r = length1_r + length2_r
        r.append(total_r)
        r_results.append((total_r, cycle1_r, cycle2_r))

        # Heurystyka rozbudowy cyklu + żal + wagi
        cycle1_wr, cycle2_wr = weighted_regret_two_cycles(distance_matrix)
        length1_wr = cycle_length(distance_matrix, cycle1_wr)
        length2_wr = cycle_length(distance_matrix, cycle2_wr)
        total_wr = length1_wr + length2_wr
        wr.append(total_wr)
        wr_results.append((total_wr, cycle1_wr, cycle2_wr))

    print("KroB200:")
    print("Nearest Neighbor: min =", min(nn), "mean =", sum(nn) / len(nn), "max =", max(nn))
    print("Greedy Cycle: min =", min(gc), "mean =", sum(gc) / len(gc), "max =", max(gc))
    print("Regret: min =", min(r), "mean =", sum(r) / len(r), "max =", max(r))
    print("Weighted Regret: min =", min(wr), "mean =", sum(wr) / len(wr), "max =", max(wr))

    # Wizualizacja dla najlepszych (najkrótszych) dróg instancji kroB200:
    best_nn = min(nn_results, key=lambda x: x[0])
    # drawCycles(coords, best_nn[1], best_nn[2], title="kroB200 - Nearest Neighbor, droga = " + str(best_nn[0]))

    best_gc = min(gc_results, key=lambda x: x[0])
    drawCycles(coords, best_gc[1], best_gc[2], title="kroB200 - Greedy Cycle, droga = " + str(best_gc[0]))

    best_r = min(r_results, key=lambda x: x[0])
    drawCycles(coords, best_r[1], best_r[2], title="kroB200 - Regret, droga = " + str(best_r[0]))

    best_wr = min(wr_results, key=lambda x: x[0])
    drawCycles(coords, best_wr[1], best_wr[2], title="kroB200 - Weighted Regret, droga = " + str(best_wr[0]))

def lab2(num):
    # Dla instancji kroA200.tsp
    filename = "kroB200.tsp"
    coords = read_tsplib(filename)

    distance_matrix = compute_distance_matrix(coords)
    n = len(distance_matrix)

    r = []
    r_results = []
    gvr = []
    gvr_results = []
    gir = []
    gir_results = []
    reg = []
    reg_results = []
    gvreg = []
    gvreg_results = []
    gireg = []
    gireg_results = []

    # Lists for tracking execution times
    times_r = []
    times_gvr = []
    times_gir = []
    times_reg = []
    times_gvreg = []
    times_gireg = []

    for _ in range(num):
        # Random method
        start_time = time.time()
        cycle1_r, cycle2_r = random_cycles(distance_matrix)
        elapsed_r = time.time() - start_time
        times_r.append(elapsed_r)

        start_time = time.time()
        cycle1_gvr, cycle2_gvr = greedy_vertice_regret(distance_matrix, cycle1_r, cycle2_r)
        elapsed_gvr = time.time() - start_time
        times_gvr.append(elapsed_gvr)

        start_time = time.time()
        cycle1_gir, cycle2_gir = greedy_inner_regret(distance_matrix, cycle1_r, cycle2_r)
        elapsed_gir = time.time() - start_time
        times_gir.append(elapsed_gir)

        # Regret method
        start_time = time.time()
        cycle1_reg, cycle2_reg = weighted_regret_two_cycles(distance_matrix)
        elapsed_reg = time.time() - start_time
        times_reg.append(elapsed_reg)

        start_time = time.time()
        cycle1_gvreg, cycle2_gvreg = greedy_vertice_regret(distance_matrix, cycle1_reg, cycle2_reg)
        elapsed_gvreg = time.time() - start_time
        times_gvreg.append(elapsed_gvreg)

        start_time = time.time()
        cycle1_gireg, cycle2_gireg = greedy_inner_regret(distance_matrix, cycle1_reg, cycle2_reg)
        elapsed_gireg = time.time() - start_time
        times_gireg.append(elapsed_gireg)

        print("wykresy in progress")

        # Random-based methods results
        length1_gvr = cycle_length(distance_matrix, cycle1_gvr)
        length2_gvr = cycle_length(distance_matrix, cycle2_gvr)
        total_gvr = length1_gvr + length2_gvr
        gvr.append(total_gvr)
        gvr_results.append((total_gvr, cycle1_gvr, cycle2_gvr))

        length1_gir = cycle_length(distance_matrix, cycle1_gir)
        length2_gir = cycle_length(distance_matrix, cycle2_gir)
        total_gir = length1_gir + length2_gir
        gir.append(total_gir)
        gir_results.append((total_gir, cycle1_gir, cycle2_gir))

        length1_r = cycle_length(distance_matrix, cycle1_r)
        length2_r = cycle_length(distance_matrix, cycle2_r)
        total_r = length1_r + length2_r
        r.append(total_r)
        r_results.append((total_r, cycle1_r, cycle2_r))

        # Regret-based methods results
        length1_gvreg = cycle_length(distance_matrix, cycle1_gvreg)
        length2_gvreg = cycle_length(distance_matrix, cycle2_gvreg)
        total_gvreg = length1_gvreg + length2_gvreg
        gvreg.append(total_gvreg)
        gvreg_results.append((total_gvreg, cycle1_gvreg, cycle2_gvreg))

        length1_gireg = cycle_length(distance_matrix, cycle1_gireg)
        length2_gireg = cycle_length(distance_matrix, cycle2_gireg)
        total_gireg = length1_gireg + length2_gireg
        gireg.append(total_gireg)
        gireg_results.append((total_gireg, cycle1_gireg, cycle2_gireg))

        length1_reg = cycle_length(distance_matrix, cycle1_reg)
        length2_reg = cycle_length(distance_matrix, cycle2_reg)
        total_reg = length1_reg + length2_reg
        reg.append(total_reg)
        reg_results.append((total_reg, cycle1_reg, cycle2_reg))


    print("===LOOP 30====")

    best_nn = min(r_results, key=lambda x: x[0])
    drawCycles(coords, best_nn[1], best_nn[2], title="kroB200 - Random, droga = " + str(best_nn[0]))

    best_gvr = min(gvr_results, key=lambda x: x[0])
    drawCycles(coords, best_gvr[1], best_gvr[2], title="kroB200 - Greedy Inter Random, droga = " + str(best_gvr[0]))

    best_gir = min(gir_results, key=lambda x: x[0])
    drawCycles(coords, best_gir[1], best_gir[2], title="kroB200 - Greedy Inner Random, droga = " + str(best_gir[0]))

    best_nn = min(reg_results, key=lambda x: x[0])
    drawCycles(coords, best_nn[1], best_nn[2], title="kroB200 - Regret, droga = " + str(best_nn[0]))

    best_gvr = min(gvreg_results, key=lambda x: x[0])
    drawCycles(coords, best_gvr[1], best_gvr[2], title="kroB200 - Greedy Inter Regret, droga = " + str(best_gvr[0]))

    best_gir = min(gireg_results, key=lambda x: x[0])
    drawCycles(coords, best_gir[1], best_gir[2], title="kroB200 - Greedy Inner Regret, droga = " + str(best_gir[0]))

    print("Random: min =", min(r), "mean =", sum(r) / len(r), "max =", max(r))
    print("Greedy vertice random: min =", min(gvr), "mean =", sum(gvr) / len(gvr), "max =", max(gvr))
    print("Greedy vertice random: min =", min(gir), "mean =", sum(gir) / len(gir), "max =", max(gir))
    print("Regret: min =", min(reg), "mean =", sum(reg) / len(reg), "max =", max(reg))
    print("Greedy vertice regret: min =", min(gvreg), "mean =", sum(gvreg) / len(gvreg), "max =", max(gvreg))
    print("Greedy vertice regret: min =", min(gireg), "mean =", sum(gireg) / len(gireg), "max =", max(gireg))

    # Print statistics
    print("\n=== Execution Time Statistics ===")
    print(
        f"Random cycles: min = {min(times_r):.4f}s, mean = {sum(times_r) / len(times_r):.4f}s, max = {max(times_r):.4f}s")
    print(
        f"Greedy Vertice Random: min = {min(times_gvr):.4f}s, mean = {sum(times_gvr) / len(times_gvr):.4f}s, max = {max(times_gvr):.4f}s")
    print(
        f"Greedy Inner Random: min = {min(times_gir):.4f}s, mean = {sum(times_gir) / len(times_gir):.4f}s, max = {max(times_gir):.4f}s")
    print(
        f"Weighted Regret Cycles: min = {min(times_reg):.4f}s, mean = {sum(times_reg) / len(times_reg):.4f}s, max = {max(times_reg):.4f}s")
    print(
        f"Greedy Vertice Regret: min = {min(times_gvreg):.4f}s, mean = {sum(times_gvreg) / len(times_gvreg):.4f}s, max = {max(times_gvreg):.4f}s")
    print(
        f"Greedy Inner Regret: min = {min(times_gireg):.4f}s, mean = {sum(times_gireg) / len(times_gireg):.4f}s, max = {max(times_gireg):.4f}s")





def main():
    lab2(10)

if __name__ == "__main__":
    main()
