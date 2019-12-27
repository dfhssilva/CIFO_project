from random import uniform, randint, choices, sample, shuffle
from copy import deepcopy

from cifo.problem.objective import ProblemObjective
from cifo.problem.solution import EncodingDataType
from cifo.problem.population import Population

###################################################################################################
# INITIALIZATION APPROACHES
###################################################################################################

# (!) REMARK:
# Initialization signature: <method_name>(problem, population_size):

# -------------------------------------------------------------------------------------------------
# Initialization Random
# -------------------------------------------------------------------------------------------------
def initialize_using_random(problem, population_size):
    """
    Initialize a population of solutions (feasible solution) for an evolutionary algorithm using Random method

    Required:
    @ problem - problem's build solution function knows how to create an individual in accordance with the encoding.
    @ population_size - to define the size of the population to be returned.
    """
    solution_list = []

    # generate a population of admissible solutions (individuals)
    for i in range(0, population_size):
        s = problem.build_solution(method='Random')
        
        # check if the solution is admissible
        while not problem.is_admissible(s):
            s = problem.build_solution(method='Random')
        
        s.id = [0, i]

        problem.evaluate_solution(s)
        
        solution_list.append(s)

    population = Population( 
        problem = problem,
        maximum_size = population_size, 
        solution_list = solution_list
    )
    
    return population

# -------------------------------------------------------------------------------------------------
# Initialization using Hill Climbing
# -------------------------------------------------------------------------------------------------
#TODO: OPTIONAL, implement a initialization based on Hill Climbing
# Remark: remember, you will need a neighborhood function for each problem
def initialize_using_hc(problem, population_size):
    """
    Initialize a population of solutions (feasible solution) for an evolutionary algorithm using Hill Climbing

    Required:
    @ problem - problem's build solution function knows how to create an individual in accordance with the encoding.
    @ population_size - to define the size of the population to be returned.
    """
    solution_list = []

    # generate a population of admissible solutions (individuals)
    for i in range(0, population_size):
        s = problem.build_solution(method='Hill Climbing')

        # check if the solution is admissible
        while not problem.is_admissible(s):
            s = problem.build_solution(method='Hill Climbing')

        s.id = [0, i]

        problem.evaluate_solution(s)

        solution_list.append(s)

    population = Population( 
        problem = problem,
        maximum_size = population_size, 
        solution_list = solution_list
    )

    return population

# -------------------------------------------------------------------------------------------------
# Initialization using Simulated Annealing
# -------------------------------------------------------------------------------------------------
#TODO: OPTIONAL, implement a initialization based on Simulated Annealing
# Remark: remember, you will need a neighborhood function for each problem
def initialize_using_sa(problem, population_size):
    """
    Initialize a population of solutions (feasible solution) for an evolutionary algorithm using Simulated Annealing

    Required:
    @ problem - problem's build solution function knows how to create an individual in accordance with the encoding.
    @ population_size - to define the size of the population to be returned.
    """
    solution_list = []

    # generate a population of admissible solutions (individuals)
    for i in range(0, population_size):
        s = problem.build_solution(method='Simulated Annealing')

        # check if the solution is admissible
        while not problem.is_admissible(s):
            s = problem.build_solution(method='Simulated Annealing')

        s.id = [0, i]

        problem.evaluate_solution(s)

        solution_list.append(s)

    population = Population(
        problem=problem,
        maximum_size=population_size,
        solution_list=solution_list
    )

    return population


# -------------------------------------------------------------------------------------------------
# Initialization using Greedy Method
# -------------------------------------------------------------------------------------------------
#TODO: OPTIONAL, implement a initialization based on Greedy Method
# Remark: remember, you will need a neighborhood function for each problem
def initialize_using_greedy(problem, population_size):
    """
    Initialize a population of solutions (feasible solution) for an evolutionary algorithm using a Greedy Algorithm

    Required:
    @ problem - problem's build solution function knows how to create an individual in accordance with the encoding.
    @ population_size - to define the size of the population to be returned.
    """
    solution_list = []

    # generate a population of admissible solutions (individuals)
    for i in range(0, population_size):
        s = problem.build_solution(method='Greedy')

        # check if the solution is admissible
        while not problem.is_admissible(s):
            s = problem.build_solution(method='Greedy')

        s.id = [0, i]

        problem.evaluate_solution(s)

        solution_list.append(s)

    population = Population(
        problem=problem,
        maximum_size=population_size,
        solution_list=solution_list
    )

    return population

###################################################################################################
# SELECTION APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# roulettewheel_selection function
# -------------------------------------------------------------------------------------------------
def roulettewheel_selection(population, objective, params): # INVESTIGAR POSSIVEIS USOS para os params, MULTIPLE ROULETTE!!!!!!!!!!!!!!!
    """
    Main idea: better individuals get higher chance
    The chances are proportional to the fitness
    Implementation: roulette wheel technique
    Assign to each individual a part of the roulette wheel
    Spin the wheel n times to select n individuals

    REMARK: This implementation does not consider minimization problem
    """
    def _select_index_max(population):
        # Get the Total Fitness (all solutions in the population) to calculate the chances proportional to fitness
        total_fitness = 0
        for solution in population.solutions:
            total_fitness += solution.fitness

        # spin the wheel
        wheel_position = uniform(0, 1)

        # calculate the position which wheel should stop
        stop_position = 0
        index = 0
        for solution in population.solutions:
            stop_position += (solution.fitness / total_fitness)
            if stop_position > wheel_position:
                break
            index += 1

        return index

    def _select_index_min(population):
        # Get the Total Fitness (all solutions in the population) to calculate the chances proportional to fitness
        total_fitness = 0
        pop_size = population.size  # get the population size for the formula

        for solution in population.solutions:
            total_fitness += solution.fitness

        # spin the wheel
        wheel_position = uniform(0, 1)

        # calculate the position which wheel should stop
        stop_position = 0
        index = 0
        for solution in population.solutions:
            stop_position += (1 - (solution.fitness / total_fitness))/(pop_size - 1)
            if stop_position > wheel_position:
                break
            index += 1

        return index

    # select two different parents using roulette wheel
    if objective == ProblemObjective.Maximization:
        index1 = _select_index_max(population=population)
        index2 = index1

        while index2 == index1:
            index2 = _select_index_max(population=population)
    elif objective == ProblemObjective.Minimization:
        index1 = _select_index_min(population=population)
        index2 = index1

        while index2 == index1:
            index2 = _select_index_min(population=population)
    else:
        print('The code does not handle multiobjective problems yet.')
        exit(code=1)

    return population.get(index1), population.get(index2)

        
# -------------------------------------------------------------------------------------------------
# rank_selection function
# -------------------------------------------------------------------------------------------------
def rank_selection(population, objective, params):
    """
    Rank Selection sorts the population first according to fitness value and ranks them. Then every chromosome is
    allocated selection probability with respect to its rank. Individuals are selected as per their selection
    probability. Rank selection is an exploration technique of selection.
    """
    def _sort(population, objective):
        if objective == ProblemObjective.Maximization: # in maximization, the solution with higher fitness is at the end
            for i in range(0, population.size):
                for j in range(i, population.size):
                    if population.solutions[i].fitness > population.solutions[j].fitness:
                        population.solutions[i], population.solutions[j] = population.solutions[j], population.solutions[i]

        else:    # in minimization, the solution with smaller fitness is at the end
            for i in range(0, population.size):
                for j in range(i, population.size):
                    if population.solutions[i].fitness < population.solutions[j].fitness:
                        population.solutions[i], population.solutions[j] = population.solutions[j], population.solutions[i]

        return population

    # Step 1: Sort / Rank
    population = _sort(population, objective)

    # Step 2: Create a rank list [0, 1, 1, 2, 2, 2, ...]
    rank_list = []

    for index in range(0, population.size):
        for _ in range(0, index + 1):
            rank_list.append(index)

    print(f" >> rank_list: {rank_list}")

    # Step 3: Select solution index
    index1 = randint(0, len(rank_list)-1)
    index2 = index1

    while index2 == index1:
        index2 = randint(0, len(rank_list)-1)

    return population.get(rank_list[index1]), population.get(rank_list[index2])

# -------------------------------------------------------------------------------------------------
# function tournament_selection
# -------------------------------------------------------------------------------------------------
def tournament_selection(population, objective, params):
    """
    """
    tournament_size = 2
    if "Tournament-Size" in params:
        tournament_size = params["Tournament-Size"]

    def _select_index(objective, population, tournament_size):
        index_temp      = -1
        index_selected  = randint(0, population.size - 1)

        if objective == ProblemObjective.Maximization:
            for _ in range(0, tournament_size):
                index_temp = randint(0, population.size - 1)

                if population.solutions[index_temp].fitness > population.solutions[index_selected].fitness:
                    index_selected = index_temp
        elif objective == ProblemObjective.Minimization:
            for _ in range(0, tournament_size):
                index_temp = randint(0, population.size - 1)

                if population.solutions[index_temp].fitness < population.solutions[index_selected].fitness:
                    index_selected = index_temp

        return index_selected


    index1 = _select_index(objective, population, tournament_size)
    index2 = index1

    while index2 == index1:
        index2 = _select_index(objective, population, tournament_size)

    return population.solutions[index1], population.solutions[index2]

# TODO: uniform selection
###################################################################################################
# CROSSOVER APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# Singlepoint crossover
# -------------------------------------------------------------------------------------------------
def singlepoint_crossover(problem, solution1, solution2):
    singlepoint = randint(0, len(solution1.representation)-1)
    #print(f" >> singlepoint: {singlepoint}")

    offspring1 = deepcopy(solution1) #solution1.clone()
    offspring2 = deepcopy(solution2) #.clone()

    for i in range(singlepoint, len(solution2.representation)):
        offspring1.representation[i] = solution2.representation[i]
        offspring2.representation[i] = solution1.representation[i]

    return offspring1, offspring2    

# -------------------------------------------------------------------------------------------------
# Partially Mapped Crossover (PMX)
# -------------------------------------------------------------------------------------------------
def pmx_crossover(problem, solution1, solution2):
    # copy the parents to the children because we only need to change the middle part and the repeated elements
    offspring1 = deepcopy(solution1)
    offspring2 = deepcopy(solution2)

    # choose the random crossover points - get two different and ordered indexes
    crosspoint1, crosspoint2 = get_two_diff_order_index(0, (len(solution1.representation) - 1))

    # change the middle part of the children
    offspring2.representation[crosspoint1:crosspoint2] = solution1.representation[crosspoint1:crosspoint2]
    offspring1.representation[crosspoint1:crosspoint2] = solution2.representation[crosspoint1:crosspoint2]

    # indexes of the elements out of the middle part (they will be equal for both solutions)
    out_index = [item for item in list(range(0, len(solution1.representation))) if
                 item not in list(range(crosspoint1, crosspoint2))]

    repeated_index1 = []            # indexes of the repeated elements solution1
    repeated_index2 = []            # indexes of the repeated elements solution2
    for i in out_index:             # get the indexes of the repeated elements
        if offspring1.representation[i] in offspring1.representation[crosspoint1:crosspoint2]:  # repeated elements
            repeated_index1.append(i)                                                           # solution 1
        if offspring2.representation[i] in offspring2.representation[crosspoint1:crosspoint2]:  # repeated elements
            repeated_index2.append(i)                                                           # solution 2


    for i in repeated_index1: # for each repeated element in offspring1
        # replace i if the element in i of the opposite solution is not in offspring1
        if solution2.representation[i] not in offspring1.representation:
            offspring1.representation[i] = solution2.representation[i]
        else:
            idx = i
            while True:
                # get the index of solution1 of the element in i of solution2
                idx = solution1.representation.index(solution2.representation[idx])

                # replace i if the element in i of the opposite solution is not in offspring1
                if solution2.representation[idx] not in offspring1.representation:
                    offspring1.representation[i] = solution2.representation[idx]
                    break

    for i in repeated_index2:   # for each repeated element in offspring2
        # replace i if the element in i of the opposite solution is not in offspring2
        if solution1.representation[i] not in offspring2.representation:
            offspring2.representation[i] = solution1.representation[i]
        else:
            idx = i
            while True:
                # get the index of solution2 of the element in i of solution1
                idx = solution2.representation.index(solution1.representation[idx])

                # replace i if the element in i of the opposite solution is not in offspring2
                if solution1.representation[idx] not in offspring2.representation:
                    offspring2.representation[i] = solution1.representation[idx]
                    break   # break the while cycle because we already finished the cycle finding the element

    return offspring1, offspring2

# -------------------------------------------------------------------------------------------------
# Cycle Crossover
# -------------------------------------------------------------------------------------------------
def cycle_crossover(problem, solution1, solution2):
    cycle = 1   # number of cycles

    # only changes when the cycle is even (considering it starts at 1)
    offspring1 = deepcopy(solution1)  # solution1.clone()
    offspring2 = deepcopy(solution2)  # .clone()

    index_visited = [] # list of visited indexes during the cycles

    while len(index_visited) < len(solution1.representation): # the loop only ends when all the indexes were visited
        for i in range(0, len(solution1.representation)): # Get the smallest index of the solution not visited
            if i not in index_visited:
                idx = i
                index_visited.append(idx)
                break

        if cycle % 2 == 0:    # when the cycle is even
            while True:
                offspring1.representation[idx] = solution2.representation[idx]    # save de swaped elements
                offspring2.representation[idx] = solution1.representation[idx]

                # get the respective index of the solution 1 for the element in solution 2
                idx = solution1.representation.index(solution2.representation[idx])

                if idx not in index_visited:    # if the index was already visited, the cycle ends
                    index_visited.append(idx)   # if not, append to the list of visited indexes
                else:
                    cycle += 1      # go to the next cycle
                    break

        else:       # when the cycle is odd
            while True:
                # get the respective index of the solution 1 for the element in solution 2
                idx = solution1.representation.index(solution2.representation[idx])

                if idx not in index_visited:    # if the index was already visited, the cycle ends
                    index_visited.append(idx)   # if not, append to the list of visited indexes
                else:
                    cycle += 1      # go to the next cycle
                    break

    return offspring1, offspring2

# -------------------------------------------------------------------------------------------------
# Order1 Crossover
# -------------------------------------------------------------------------------------------------
def order1_crossover(problem, solution1, solution2):
    # copy the parents to the children because we only need to change the middle part and the repeated elements
    offspring1 = deepcopy(solution1)
    offspring2 = deepcopy(solution2)

    # choose the random crossover points
    crosspoint1, crosspoint2 = get_two_diff_order_index(0, (len(solution1.representation) - 1))  # get two different, ordered, indexes

    # indexes of the elements not in the middle
    sub = [*range((crosspoint2+1), len(offspring1.representation))]+[*range(0, crosspoint1)]
    j = 0
    k = 0

    if crosspoint2 == (len(solution1.representation)-1) and crosspoint1 == 0:
        return solution1, solution2
    else:
                        # order by which elements must be considered
        for i in [*range((crosspoint2+1), len(offspring1.representation))]+[*range(0, (crosspoint2+1))]:

            # replace offspring1 if element is not in the middle
            if solution2.representation[i] not in solution1.representation[crosspoint1:(crosspoint2+1)]:
                offspring1.representation[sub[j]] = solution2.representation[i]
                j += 1

            # replace offspring2 if element is not in the middle
            if solution1.representation[i] not in solution2.representation[crosspoint1:(crosspoint2+1)]:
                offspring2.representation[sub[k]] = solution1.representation[i]
                k += 1

            if j == len(sub) and k == len(sub):
                break

        return offspring1, offspring2

# TODO: use more than one crossovers in the same run
# TODO: heuristic crossover
# TODO: edge crossover
###################################################################################################
# MUTATION APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# Singlepoint mutation
# -----------------------------------------------------------------------------------------------
def single_point_mutation(problem, solution):
    singlepoint = randint(0, len( solution.representation )-1)
    #print(f" >> singlepoint: {singlepoint}")

    encoding    = problem.encoding

    if encoding.encoding_type == EncodingDataType.choices :
        try:
            temp = deepcopy(encoding.encoding_data)

            temp.pop(solution.representation[singlepoint])

            gene = temp[0]
            if len(temp) > 1 : gene = choices(temp)

            solution.representation[singlepoint] = gene

            return solution
        except:
            print('(!) Error: singlepoint mutation encoding.data issues)')

    # return solution           

# -------------------------------------------------------------------------------------------------
# Swap mutation
# -----------------------------------------------------------------------------------------------
def swap_mutation(problem, solution):
    index = sample(range(0,len(solution.representation)), 2)

    solution.representation[index[0]], solution.representation[index[1]] = solution.representation[index[1]], solution.representation[index[0]]

    return solution

# -------------------------------------------------------------------------------------------------
# Insert mutation
# -----------------------------------------------------------------------------------------------
def insert_mutation(problem, solution):
    solution2 = deepcopy(solution)

    mutpoint1, mutpoint2 = get_two_diff_order_index(0,(len(solution.representation) - 1)) # get two indexes

    # inserting the second value in the index after 1st mutation point
    solution2.representation[(mutpoint1 + 1)] = solution.representation[mutpoint2]

    # passing the rest of the elements by their original order to after the number inserted
    for i in range((mutpoint1 + 2), (mutpoint2 + 1)):
        solution2.representation[i] = solution.representation[(i - 1)]

    return solution2

# -------------------------------------------------------------------------------------------------
# Inversion mutation
# -----------------------------------------------------------------------------------------------
def inversion_mutation(problem, solution):
    solution2 = deepcopy(solution) # create a copy that we will edit and return

    mutpoint1, mutpoint2 = get_two_diff_order_index(0,(len(solution.representation) - 1)) # get two indexes

    for i in range(mutpoint1, (mutpoint2 + 1)): # replace the indexes by inverse order
        solution2.representation[i] = solution.representation[mutpoint2]
        mutpoint2 -= 1

    return solution2

# -------------------------------------------------------------------------------------------------
# Scramble mutation
# -----------------------------------------------------------------------------------------------
def scramble_mutation(problem, solution):

    mutpoint1, mutpoint2 = get_two_diff_order_index(0, (len(solution.representation) - 1))  # get two indexes

    shuffle_part = solution.representation[mutpoint1: (mutpoint2+1)] #store the part of the solution to be shuffled in separate list

    shuffle(shuffle_part) #shuffle the middle part (shuffle works inplace)

    solution.representation[mutpoint1: (mutpoint2 + 1)] = shuffle_part #replace the segment with shuffled part

    return solution


#TODO: Implement Greedy Swap mutation
###################################################################################################
# REPLACEMENT APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# Standard replacement
# -----------------------------------------------------------------------------------------------
def standard_replacement(problem, current_population, new_population):
    return deepcopy(new_population)

# -------------------------------------------------------------------------------------------------
# Elitism replacement
# -----------------------------------------------------------------------------------------------
def elitism_replacement(problem, current_population, new_population):
    if problem.objective == ProblemObjective.Minimization:
        if current_population.fittest.fitness < new_population.fittest.fitness:
           new_population.solutions[0] = current_population.solutions[-1]

    elif problem.objective == ProblemObjective.Maximization:
        if current_population.fittest.fitness > new_population.fittest.fitness:
           new_population.solutions[0] = current_population.solutions[-1]

    return deepcopy(new_population)

###################################################################################################
# HELPER FUNCTIONS
####################################################################################################
def get_two_diff_order_index(start=0, stop=1, order=True, diff=True):
    """
    Returns two integers from a range, they can be:
        put in order (default) or unordered
        always different(default) or can be repeated
    start - integer (default = 0)
    stop - integer (default= 1)
    order - boolean ( default= True)
    """
    first = randint(start, stop)
    second = randint(start, stop)

    if diff:
        while first == second:
            second = randint(start, stop)
    if order:
        if first > second:
            second, first = first, second

    return first, second
