import math
import random

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


def main():

    filename = "./lab1/kroA200.tsp"
    coords = read_tsplib(filename)
    
    distance_matrix = compute_distance_matrix(coords)
    n = len(distance_matrix)
    
    # Heurystyka najbliższego sąsiada
    cycle1_nn, cycle2_nn = nearest_neighbor_two_cycles(distance_matrix)
    length1_nn = cycle_length(distance_matrix, cycle1_nn)
    length2_nn = cycle_length(distance_matrix, cycle2_nn)
    total_nn = length1_nn + length2_nn
    
    print("== Heurystyka Najbliższego Sąsiada ==")
    print("Cykl 1:", cycle1_nn)
    print("Długość cyklu 1:", length1_nn)
    print("Cykl 2:", cycle2_nn)
    print("Długość cyklu 2:", length2_nn)
    print("Suma długości:", total_nn)
    
    # Heurystyka rozbudowy cyklu
    cycle1_gc, cycle2_gc = greedy_cycle_two_cycles(distance_matrix)
    length1_gc = cycle_length(distance_matrix, cycle1_gc)
    length2_gc = cycle_length(distance_matrix, cycle2_gc)
    total_gc = length1_gc + length2_gc
    
    print("\n== Heurystyka Rozbudowy Cyklu (Greedy Cycle) ==")
    print("Cykl 1:", cycle1_gc)
    print("Długość cyklu 1:", length1_gc)
    print("Cykl 2:", cycle2_gc)
    print("Długość cyklu 2:", length2_gc)
    print("Suma długości:", total_gc)


if __name__ == "__main__":
    main()
