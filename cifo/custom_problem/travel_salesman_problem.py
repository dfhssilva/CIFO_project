from copy import deepcopy
from random import choice, randint, shuffle
import numpy as np

from cifo.problem.problem_template import ProblemTemplate
from cifo.problem.objective import ProblemObjective
from cifo.problem.solution import LinearSolution, Encoding

tsp_encoding_rule = {
    "Size"         : -1, # Number of the cities from the distance matrix
    "Is ordered"   : True,
    "Can repeat"   : False,
    "Data"         : [0,0], # must be defined by the data
    "Data Type"    : "Pattern"
}

# Default Matrix
input = [
        [0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
        [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
        [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
        [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
        [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],
        [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],
        [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701],
        [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099],
        [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],
        [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],
        [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200],
        [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504],
        [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0],
    ]

tsp_decision_variables_example = {
    "Distances" : np.array(input), #<< Number, Mandatory
    "Item-Name" : [i for i in range(1,len(input[0]))] # << String, Optional, The names of the items, can be used to
                                                        # present the solution.
        # It starts at 1 reserving the 0 for the static city
        # We are setting the departure point, so we only have to arrange the 12 remaining cities
}

# REMARK: There is no constraint

 # -------------------------------------------------------------------------------------------------
# TSP - Travel Salesman Problem
# -------------------------------------------------------------------------------------------------
class TravelSalesmanProblem(ProblemTemplate):
    """
    Travel Salesman Problem: Given a list of cities and the distances between each pair of cities, what is the shortest
    possible route that visits each city and returns to the origin city?
    """

    # Constructor
    #----------------------------------------------------------------------------------------------
    def __init__(self, decision_variables = tsp_decision_variables_example, constraints = None ,
                 encoding_rule = tsp_encoding_rule):
        """
        - Defines:
            - the Size and the enconding characters (Data) according to the input distances matrix
            - the objective type (Max or Min)
            - the problem name
        - Optimizes the access to the decision variables
        """
        # optimize the access to the decision variables
        self._distances = np.array([])
        if "Distances" in decision_variables:
            self._distances = decision_variables["Distances"]

        encoding_rule["Size"] = len(self._distances) - 1
        encoding_rule["Data"] = decision_variables["Item-Name"]

        # Call the Parent-class constructor to store these values and to execute any other logic to be implemented by
        # the constructor of the super-class
        super().__init__(
            decision_variables = decision_variables, 
            constraints = constraints, 
            encoding_rule = encoding_rule
        )

        # 1. Define the Name of the Problem
        self._name = "Travel Salesman Problem"
        
        # 2. Define the Problem Objective
        self._objective = ProblemObjective.Minimization

    # Build Solution for Travel Salesman Problem
    #----------------------------------------------------------------------------------------------
    def build_solution(self, method = 'Random'):
        """
        Creates a initial solution and returns it according to the enconding rule and the chosen method from the
        following available methods:
            - 'Hill Climbing'
            - 'Simulated Annealing'
            - 'Random' (default method)
            - 'Greedy'
        """
        if method == 'Random': # shuffles the list and then instanciates it as a Linear Solution
            encoding_data = self._encoding.encoding_data

            solution_representation = encoding_data.copy()
            np.random.shuffle(solution_representation) # inplace suffle

            solution = LinearSolution(
                representation = solution_representation,
                encoding_rule = self._encoding_rule
            )

            return solution
        else:
            print('Please choose one of these methods: Hill Climbing, Simulated Annealing or Random. It will use the '
                  'Random method')
            return self.build_solution()

    
    # Solution Admissibility Function - is_admissible()
    #----------------------------------------------------------------------------------------------
    def is_admissible(self, solution, debug=False): #<< use this signature in the sub classes, the meta-heuristic 
        """
        Checks if the solution:
            - has unique values equal to the size of the enconding_rule
            - has length equal to the size of the enconding_rule
            - has all values within the defined range of item-names
        If all these conditions are true, the solution is admissible and it returns True.
        """
        if debug:
            result = False

            if (len(set(solution)) == self.encoding_rule["Size"]) & (len(solution) == self.encoding_rule["Size"]) & \
                    all([True if i in range(1,(self.encoding_rule["Size"]+1)) else False for i in solution]):
                result = True

            print(solution, result)

            return result
        else:
            return True

    # Evaluate_solution()
    #-------------------------------------------------------------------------------------------------------------
    # It should be seen as an abstract method 
    def evaluate_solution(self, solution, feedback = None): # << This method does not need to be extended, it already
                                        # automated solutions evaluation, for Single-Objective and for Multi-Objective
        """
        Calculates the total distance of the defined route
        """
        distances = self._distances
        rep = solution.representation

        fitness = distances[0, rep[0]] + distances[rep[-1], 0]  # the distances are assymmetric
            # distance from the departure point (position 0 in the matrix) to the first city(position 0 in the solution)
            # plus
            # distance from the last city to the departure point (position 0 in the solution)

        for i in range(0, (len(rep)-1)):
            fitness += distances[rep[i], rep[i+1]]

        solution._fitness = fitness
        solution._is_fitness_calculated = True

        return solution    

# -------------------------------------------------------------------------------------------------
# OPTIONAL - it is only needed if you will implement Local Search Methods
#            (Hill Climbing and Simulated Annealing)
# -------------------------------------------------------------------------------------------------
def tsp_bitflip_get_neighbors(solution, problem, neighborhood_size = 0):

    pass