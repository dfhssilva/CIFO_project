# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------------------

from cifo.algorithm.genetic_algorithm import GeneticAlgorithm
from cifo.algorithm.hill_climbing import HillClimbing
from cifo.custom_problem.travel_salesman_problem import (
    TravelSalesmanProblem, tsp_decision_variables_example, tsp_get_neighbors
)
from cifo.problem.objective import ProblemObjective
from cifo.algorithm.ga_operators import (
    initialize_using_random, initialize_using_hc, initialize_using_sa, initialize_using_greedy, initialize_using_multiple,
    roulettewheel_selection, rank_selection, tournament_selection,
    singlepoint_crossover, cycle_crossover, pmx_crossover, order1_crossover, heuristic_crossover, multiple_crossover,
    single_point_mutation, swap_mutation, insert_mutation, inversion_mutation, scramble_mutation, greedy_mutation,
    multiple_mutation,
    elitism_replacement, standard_replacement 
)    
from cifo.util.terminal import Terminal, FontColor
from cifo.util.observer import LocalSearchObserver
from random import randint
import numpy as np
from copy import deepcopy

from os import listdir, path, mkdir
from os.path import isfile, join
from pandas import pandas as pd

def plot_performance_chart(df):
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    x = df["Generation"] 
    x_rev = x[::-1]
    y1 = df["Fitness_Mean"] 
    y1_upper = df["Fitness_Lower"]
    y1_lower = df["Fitness_Upper"]

    # line
    trace1 = go.Scatter(
        x = x,
        y = y1,
        line=dict(color='rgb(0,100,80)'),
        mode='lines',
        name='Fair',
    )

    trace2 = go.Scatter(
        x = x,
        y = y1_upper,
        fill='tozerox',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='Fair',
    )

    trace3 = go.Scatter(
        x = x,
        y = y1_lower,
        fill='tozerox',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='Fair',
    )

    data = [trace1]
    
    layout = go.Layout(
        paper_bgcolor='rgb(255,255,255)',
        plot_bgcolor='rgb(229,229,229)',
        xaxis=dict(
            gridcolor='rgb(255,255,255)',
            range=[1,10],
            showgrid=True,
            showline=False,
            showticklabels=True,
            tickcolor='rgb(127,127,127)',
            ticks='outside',
            zeroline=False
        ),
        yaxis=dict(
            gridcolor='rgb(255,255,255)',
            showgrid=True,
            showline=False,
            showticklabels=True,
            tickcolor='rgb(127,127,127)',
            ticks='outside',
            zeroline=False
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    fig.show()

# Problem
#--------------------------------------------------------------------------------------------------
# Decision Variables
input =[
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

tsp_decision_variables = {
    "Distances" : np.array(input), #<< Number, Mandatory
    "Item-Name" : [i for i in range(1, len(input[0]))] # << String, Optional, The names of the items, can be used to present the solution.
        # It starts at 1 reserving the 0 for the static city
        # We are setting the departure point, so we only have to arrange the n-1 remaining cities
}

# Problem Instance
tsp_problem_instance = TravelSalesmanProblem(
    decision_variables=tsp_decision_variables
)

# Configuration
#--------------------------------------------------------------------------------------------------
# parent selection object
# parent_selection = TournamentSelection()
parent_selection = roulettewheel_selection

# dictionaries to create test directories
valid_Init = {initialize_using_random: "rand", initialize_using_hc: "hc", initialize_using_greedy: "greedyI",
              initialize_using_multiple: "mixI"} #, initialize_using_sa: "sa"

valid_Select = {roulettewheel_selection: "rol", tournament_selection: "tourn", rank_selection: "rank"}

valid_Xover = {cycle_crossover: "cycle", pmx_crossover: "pmx",  order1_crossover: "order1", heuristic_crossover: "heur",
               multiple_crossover: "mixC"}
                # singlepoint_crossover: "singP" should not be used
valid_Mutation = {swap_mutation: "swap", insert_mutation: "insert", inversion_mutation: "invert",
                  scramble_mutation: "scramble",  greedy_mutation: "greedyM", multiple_mutation: "mixM"}
                    # single_point_mutation: "singP" should not be used

valid_Replacement = {elitism_replacement: "elit", standard_replacement: "std"}


#Parameters to gridsearch in a run
test_init = [initialize_using_multiple, initialize_using_hc, initialize_using_greedy, initialize_using_random]
test_select = [roulettewheel_selection, tournament_selection, rank_selection]
test_xover = [multiple_crossover, heuristic_crossover, cycle_crossover, pmx_crossover, order1_crossover] # singlepoint_crossover should not be used
test_mutation = [multiple_mutation, greedy_mutation, swap_mutation, insert_mutation, inversion_mutation, scramble_mutation] # single_point_mutation should not be used
test_replacement = [elitism_replacement] #standard_replacement
#test_xover_prob = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
test_xover_prob = [0.9, 0.1]
test_mut_prob = [0.9, 0.1]
#test_xover_prob = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
test_tournament_size = [2, 5, 10]
#test_tournament_size = [2] # simples


#initial params

params = {
        # params
        "Population-Size"           : 40,
        "Number-of-Generations"     : 1000,
        "Crossover-Probability"     : 0.8,
        "Mutation-Probability"      : 0.8,
        # operators / approaches
        "Initialization-Approach"   : initialize_using_random,
        "Selection-Approach"        : parent_selection,
        "Tournament-Size"           : 5,
        "Crossover-Approach"        : cycle_crossover,
        "Mutation-Approach"         : swap_mutation,
        "Replacement-Approach"      : elitism_replacement
    }


def one_combination():
    """
    Actually runs the algorithm with one set of parameters.
    Names the resume file from parameters of search
    Creates the resume file from all the runs for a set of parameters

    """

    log_base_dir="./log/"       #Base dir for log of initial runs
    if not path.exists(log_base_dir):
        mkdir(log_base_dir)
    all_dir = "./log_all/"      #Base dir for resume the resume files
    if not path.exists(all_dir):
        mkdir(all_dir)

    log_name =  ("I-"    + str(valid_Init.get(params.get("Initialization-Approach"))) +
                "_S-"    + str(valid_Select.get(params.get("Selection-Approach"))) +
                "_C-"    + str(valid_Xover.get(params.get("Crossover-Approach"))) +
                "_M-"    + str(valid_Mutation.get(params.get("Mutation-Approach"))) +
                "_R-"    + str(valid_Replacement.get(params.get("Replacement-Approach"))) +
                "_CP-"   + str((params.get("Crossover-Probability"))) +
                "_MP-"   + str((params.get("Mutation-Probability"))) +
                "_PS-"   + str((params.get("Population-Size"))) +
                "_TS-"   + str((params.get("Tournament-Size"))) +
                "_G-"    + str((params.get("Number-of-Generations")))
                )
    resume_name= f"{all_dir}{log_name}.xlsx"

    # checks if the same run as already been performed, if so, skips it.
    # -----------------
    log_dir = str(log_base_dir) + str(log_name)
    if not path.exists(log_dir):
       mkdir(log_dir)

    runs_made = [f for f in listdir(all_dir) if isfile(join(all_dir, f))]
    if (str(log_name)+".xlsx") in runs_made:
        print (f"Run {log_name}.xlsx already made, skipping")
        return

    # Run the same configuration many times (get distribution)
    #--------------------------------------------------------------------------------------------------
    overall_best_solution = None
    number_of_runs = 30
    for run in range(1, number_of_runs + 1):
        # Genetic Algorithm
        ga = GeneticAlgorithm(
            problem_instance=tsp_problem_instance,
            params=params,
            run=run,
            log_name=log_name,
            log_dir=log_base_dir
            )

        ga_observer = LocalSearchObserver(ga)
        ga.register_observer(ga_observer)
        ga.search()
        ga.save_log()

        # find the best solution over the runs
        if run == 1:
            overall_best_solution = deepcopy(ga.best_solution)
        else:
            if ga.best_solution.fitness < overall_best_solution.fitness:
                overall_best_solution = deepcopy(ga.best_solution)

        print('overall_best_solution: ', overall_best_solution.representation)
        print('overall_best_solution fitness: ', overall_best_solution.fitness)

    # Consolidate the runs
    #--------------------------------------------------------------------------------------------------

    log_files = [f for f in listdir(log_dir) if isfile(join(log_dir, f))]
    print(log_files)

    fitness_runs = []
    columns_name = []
    counter = 0
    generations = []

    for log_name in log_files:
        if log_name.startswith('run_'):
            df = pd.read_excel(log_dir + "/" + log_name)
            fitness_runs.append(list(df.Fitness))
            columns_name.append(log_name.strip(".xslx"))
            counter += 1

            if not generations:
                generations = list(df["Generation"])

    #fitness_sum = [sum(x) for x in zip(*fitness_runs)]

    df = pd.DataFrame(list(zip(*fitness_runs)), columns=columns_name)

    fitness_sd   = list(df.std(axis=1))
    fitness_mean = list(df.mean(axis=1))

    #df["Fitness_Sum"] = fitness_sum
    df["Generation"]  = generations
    df["Fitness_SD"]  = fitness_sd
    df["Fitness_Mean"]  = fitness_mean
    df["Fitness_Lower"] = df["Fitness_Mean"] - 1.96 * df["Fitness_SD"] / (number_of_runs ** 0.5)
    df["Fitness_Upper"] = df["Fitness_Mean"] + 1.96 * df["Fitness_SD"] / (number_of_runs ** 0.5)

    #df.to_excel(log_dir + "/all.xlsx", index=False, encoding='utf-8')
    log_name = ("I-" + str(valid_Init.get(params.get("Initialization-Approach"))) +
                "_S-" + str(
                valid_Select.get(params.get("Selection-Approach"))) +  # this one return None because of .select method
                "_C-" + str(valid_Xover.get(params.get("Crossover-Approach"))) +
                "_M-" + str(valid_Mutation.get(params.get("Mutation-Approach"))) +
                "_R-" + str(valid_Replacement.get(params.get("Replacement-Approach"))) +
                "_CP-" + str((params.get("Crossover-Probability"))) +
                "_MP-" + str((params.get("Mutation-Probability"))) +
                "_PS-" + str((params.get("Population-Size"))) +
                "_TS-" + str((params.get("Tournament-Size"))) +
                "_G-" + str((params.get("Number-of-Generations")))
                )

    # Exporting summary of configuration with best solution
    with pd.ExcelWriter(all_dir + f"{log_name}.xlsx") as writer:
        df.to_excel(writer, sheet_name='Fitness', index=False, encoding='utf-8')
        pd.DataFrame([[overall_best_solution.representation, overall_best_solution.fitness]],
                     columns=["Representation", "Fitness"]).to_excel(writer, sheet_name='Overall_Best_Solution')


#plot_performance_chart(df)

# -------------------------------------------------------------------------------------------------
# RUNS Through the Several list of possible parameters
# -------------------------------------------------------------------------------------------------
for init in range(len(test_init)):
    params["Initialization-Approach"] = test_init[init]
    for select in range(len(test_select)):
        params["Selection-Approach"] = test_select[select]
        for xover in range(len(test_xover)):
            params["Crossover-Approach"] = test_xover[xover]
            for mutation in range(len(test_mutation)):
                params["Mutation-Approach"] = test_mutation[mutation]
                for replacement in range(len(test_replacement)):
                    params["Replacement-Approach"] = test_replacement[replacement]
                    for xover_prob in range(len(test_xover_prob)):
                        params["Crossover-Probability"] = test_xover_prob[xover_prob]
                        for mut_prob in range(len(test_mut_prob)):
                            params["Mutation-Probability"] = test_mut_prob[mut_prob]
                            if str(test_select[select]) == str(tournament_selection):
                                for tourn_size in range(len(test_tournament_size)):
                                    params["Tournament-Size"] = test_tournament_size[tourn_size]
                                    one_combination()
                            else:
                                one_combination()

#[sum(sublist) for sublist in itertools.izip(*myListOfLists)]



