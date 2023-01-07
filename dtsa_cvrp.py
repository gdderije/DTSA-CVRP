import math
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
opt2 = __import__('opt2')


# distance function computes the distance between two points
def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# totalDistance function computes the total distance of the given Hamiltonian cycle
def totalDistance(tree, distance_matrix):
    distance = 0
    for i in range(len(tree)):
        distance += distance_matrix[tree[i]][tree[(i + 1) % len(tree)]]
    return distance


# nearestNeighborTour is a greedy algorithm which generates
# a Hamiltonian cycle by choosing the closest neighbor to the previous vertex
def nearestNeighborTour(distance_matrix, num_vertices):
    remaining = [i for i in range(num_vertices)]
    tour = []
    start = random.choice(remaining)
    tour.append(remaining.pop(start))
    while len(remaining) != 0:
        min_distance = distance_matrix[tour[-1]][remaining[0]]
        index = 0
        for i in range(1, len(remaining)):
            dist = distance_matrix[tour[-1]][remaining[i]]
            if dist < min_distance:
                min_distance = dist
                index = i
        tour.append(remaining.pop(index))
    return sorted(tour, key=lambda x: float(x), reverse=True)


# initializeTrees generates the initial population of trees (solutions)
def initializeTrees(distance_matrix, num_vertices, population_size):
    indices = [i for i in range(num_vertices)]
    trees = []
    trees.append(nearestNeighborTour(distance_matrix, num_vertices))
    for i in range(population_size - 1):
        tree = random.sample(indices, num_vertices)
        trees.append(tree)
    return trees


# swap transformation operator
def swap(input_tree):
    tree = [vertex for vertex in input_tree]
    index = random.sample(list(range(len(tree))), k=2)
    tree[index[0]], tree[index[1]] = tree[index[1]], tree[index[0]]
    return tree


# shift transformation operator
def shift(input_tree):
    tree = [vertex for vertex in input_tree]
    index = random.sample(list(range(len(tree))), k=2)
    while max(index) - min(index) == 1:
        index = random.sample(list(range(len(tree))), k=2)
    left_index = min(index)
    right_index = max(index)
    temp = tree[left_index]
    for i in range(left_index, right_index):
        tree[i] = tree[i + 1]
    tree[right_index] = temp
    return tree


# symmetry transformation operator
def symmetry(input_tree):
    tree = [vertex for vertex in input_tree]
    size = random.randint(2, len(tree) // 2 - 2)
    block1_start = random.choice([x for x in range(len(tree))])
    block2_start = random.choice([x for x in range(len(tree)) if x not in [y % len(tree) for y in
                                                                           range(block1_start - size + 1,
                                                                                 block1_start + size)]])
    # while block2_start >= block1_start - size + 1 and block2_start <= block1_start + size - 1:
    # 	block2_start = random.randint(0, len(tree) - 1)
    new_tree = [vertex for vertex in input_tree]
    for i in range(size):
        new_tree[(block1_start + i) % len(tree)] = tree[(block2_start + size - 1 - i) % len(tree)]
        new_tree[(block2_start + i) % len(tree)] = tree[(block1_start + size - 1 - i) % len(tree)]
    return new_tree


# implementation of the main algorithm
def DTSA(distance_matrix, num_vertices, population_size, search_tendency):
    max_FE = population_size * 2000
    FE = 0
    trees = initializeTrees(distance_matrix, num_vertices, population_size)
    distances = [totalDistance(tree, distance_matrix) for tree in trees]
    FE += population_size
    best_tree_index = distances.index(min(distances))
    while FE < max_FE:
        new_trees = []
        new_distances = []
        for i in range(population_size):
            current_tree = trees[i]
            best_tree = trees[best_tree_index]
            random_tree = trees[random.choice([x for x in range(len(trees)) if x != i and x != best_tree_index])]
            seeds = []
            if random.random() <= search_tendency:
                seeds.append(swap(best_tree))
                seeds.append(shift(best_tree))
                seeds.append(symmetry(best_tree))
                seeds.append(swap(random_tree))
                seeds.append(shift(random_tree))
                seeds.append(symmetry(random_tree))
            else:
                seeds.append(swap(current_tree))
                seeds.append(shift(current_tree))
                seeds.append(symmetry(current_tree))
                seeds.append(swap(random_tree))
                seeds.append(shift(random_tree))
                seeds.append(symmetry(random_tree))
            seed_distances = [totalDistance(seed, distance_matrix) for seed in seeds]
            FE += 6
            best_seed_distance = min(seed_distances)
            best_seed = seeds[seed_distances.index(best_seed_distance)]
            if best_seed_distance < distances[i]:
                new_trees.append(best_seed)
                new_distances.append(best_seed_distance)
            else:
                new_trees.append(trees[i])
                new_distances.append(distances[i])
        trees = [tree for tree in new_trees]
        distances = [distance for distance in new_distances]
        best_tree_index = distances.index(min(distances))
    return trees, distances, best_tree_index


def routeToSubroute(routes, customer_demand):
    """
    Inputs: Sequence of customers that a route has
            loaded instance problem
    Outputs: Route that is divided into subroutes
            which is assigned to each vehicle
    """
    route = []
    sub_route = []
    vehicle_load = 0
    vehicle_capacity = 400
    last_customer_id = 0
    for customer_id in routes:
        demand = customer_demand[customer_id]
        updated_vehicle_load = vehicle_load + demand
        if demand != 0:
            if updated_vehicle_load <= vehicle_capacity:
                sub_route.append(customer_id)
                vehicle_load = updated_vehicle_load
            else:
                route.append(sub_route)
                sub_route = [customer_id]
                vehicle_load = demand
        last_customer_id = customer_id
    if sub_route != []:
        route.append(sub_route)
    return route


def printRoute(routes, merge=False):
    route_str = '0'
    sub_route_count = 0
    for sub_route in routes:
        sub_route_count += 1
        sub_route_str = '0'
        for customer_id in sub_route:
            sub_route_str = f'{sub_route_str} - {customer_id}'  # {sub_route_str} - {customer_id}
            route_str = f'{route_str} - {customer_id}'
        sub_route_str = f'{sub_route_str} - 0'
        if not merge:
            print(f'  Vehicle {sub_route_count}\'s route: {sub_route_str}')
        route_str = f'{route_str} - 0'
    if merge:
        print(route_str)


def getNumVehiclesRequired(updated_route):
    num_of_vehicles = len(updated_route)
    return num_of_vehicles


def getRouteCost(updated_route, distance_matrix, unit_cost=1):
    """
    Inputs:
        - Individual route
        - Demands
        - Unit cost for the route (can be petrol etc.)
    Outputs:
        - Total cost for the route taken by all the vehicles
    """
    total_cost = 0
    for sub_route in updated_route:
        # Initializing the subroute distance to 0
        sub_route_distance = 0
        # Initializing customer id for depot as 0
        last_customer_id = 0
        for customer_id in sub_route:
            # Distance from the last customer id to next one in the given subroute
            distance = distance_matrix[last_customer_id][customer_id]
            sub_route_distance = sub_route_distance + distance
            # Update last_customer_id to the new one
            last_customer_id = customer_id
        # After adding distances in subroute, adding the route cost from last customer to depot that is 0
        sub_route_distance = sub_route_distance + distance_matrix[last_customer_id][0]
        # Cost for this particular subroute
        sub_route_transport_cost = unit_cost * sub_route_distance
        # Adding this to total cost
        total_cost = total_cost + sub_route_transport_cost
    return total_cost


def eval_individual_fitness(updated_route, distance_matrix, customer_demand, unit_cost):
    """
    Inputs: Individual route as a sequence
            Demand
            Unit_cost for the distance
    Outputs: Return a tuple of (Number of vehicles, route cost from all the vehicles
    """
    vehicles = getNumVehiclesRequired(updated_route)
    route_cost = getRouteCost(updated_route, distance_matrix, unit_cost)
    return vehicles, route_cost


def plotSubroute(subroute, data, color):
    totalSubroute = [0] + subroute + [0]
    subroutelen = len(subroute)
    for i in range(subroutelen + 1):
        firstcust = totalSubroute[0]
        secondcust = totalSubroute[1]
        plt.plot([data.x[firstcust], data.x[secondcust]], [data.y[firstcust], data.y[secondcust]], c=color)
        totalSubroute.pop(0)


def plotRoute(trees, num_vertices, data,
              directory, filename
              ):
    colorslist = ["#de6262", "#dea062", "#c3de62", "#94de62", "#62dea2",
                  "#62dade", "#627fde", "#a862de", "#d862de", "#de62a8",
                  "#de6275", "#8f0b0b", "#8f4d0b", "#778f0b", "#0b8f47"]
    colorindex = 0
    for i in range(num_vertices):
        if i == 0:
            plt.scatter(data.x[i], data.y[i], c='g', s=50)
            plt.text(data.x[i], data.y[i], "depot", fontsize=12)
        else:
            plt.scatter(data.x[i], data.y[i], c='b', s=50)
            plt.text(data.x[i], data.y[i], f'{i}', fontsize=12)
    for route in trees:
        plotSubroute(route, data, color=colorslist[colorindex])
        colorindex += 1
        plt.savefig(directory + filename, dpi=300)
    # plt.show()
    plt.close()


def solveVRP(save_graphs=True):
    VRP = [("P-n101-k4", 681)]
    directory = "dataset/"
    for problem, optimal_distance in VRP:
        print("Solving %s using TSA" % problem)
        print("-" * 100)
        print("VRP: %s" % (problem))
        print("Optimal Solution: %.2f" % (optimal_distance))
        file = open(directory + problem + ".vrp", "r")
        file1 = open(directory + problem + ".vrp", "r")
        file2 = open(directory + problem + ".vrp", "r")
        depots = file.readlines()[7:8]
        points = file1.readlines()[7:108]
        demand_points = file2.readlines()[109:210]
        depot = []
        vertices = []
        demands = []
        for depot_point in depots:
            depot_values = depot_point.split()
            depot.append((float(depot_values[1]), float(depot_values[2])))
        depot_x, depot_y = zip(*depot)
        for point in points:
            point_values = point.split()
            vertices.append((float(point_values[1]), float(point_values[2])))
        x, y = zip(*vertices)
        for demand_point in demand_points:
            demand_values = demand_point.split()
            demands.append((float(demand_values[0]), float(demand_values[1])))
        customer_id, demand = zip(*demands)
        num_vertices = len(vertices)
        data = pd.DataFrame({"x": x, "y": y})
        print(f'Depot coordinates: {str(depot)}')
        print("Vertices:")
        print(f'\t{"Vertex" : ^6}{"Coordinate" : ^15}{"Demand" : ^5}')
        for i in range(num_vertices):
            print(f'\t{i : ^6}{str(vertices[i]) : ^15}{str(demand[i]) : ^5}')
        distance_matrix = [[0 for i in range(num_vertices)] for j in range(num_vertices)]
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                distance_matrix[i][j] = distance(vertices[i], vertices[j])
                distance_matrix[j][i] = distance(vertices[i], vertices[j])
        population_size = num_vertices
        max_FE = population_size * 2000
        search_tendency = 0.5
        final_distances = []
        errors = []
        print("Results:")
        print(f'\t{"Trial" : ^5}{"Distance" : ^15}{"Relative Error (%)" : ^20}{"Number of Vehicles" : ^25}')
        for trial in range(30):
            solutions, distances, best_tree_index = DTSA(distance_matrix, num_vertices, population_size,
                                                         search_tendency)
            two_opt, two_opt_distances = opt2.LSFast(solutions[best_tree_index], distances[best_tree_index],
                                                     distance_matrix)
            final_route = routeToSubroute(two_opt, demand)
            final_distance = round(getRouteCost(final_route, distance_matrix), 2)
            error = abs((optimal_distance - final_distance) / optimal_distance) * 100
            eval_fitness = eval_individual_fitness(final_route, distance_matrix, demand, 1)
            final_distances.append(final_distance)
            errors.append(error)
            print(f'\t{trial + 1 : ^5}{final_distance : ^15.2f}{error : ^20.2f}{eval_fitness[0] : ^24}')
            if save_graphs:
                plotRoute(final_route, num_vertices, data, "results/",
                          f'{problem}_search tendency={search_tendency}_trial{trial + 1}.png')
        mean, stdev, error = np.mean(final_distances), np.std(final_distances), np.mean(errors)
        print("Summary:")
        print("\tMean: %.2f" % mean)
        print("\tStandard Deviation: %.2f" % stdev)
        print(f"\tError: {error : .2f}%")


# Driver program
solveVRP()
