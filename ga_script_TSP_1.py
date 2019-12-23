# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------------------

from cifo.algorithm.genetic_algorithm import GeneticAlgorithm
from cifo.algorithm.hill_climbing import HillClimbing
from cifo.custom_problem.travel_salesman_problem import (
    TravelSalesmanProblem, tsp_decision_variables_example, tsp_bitflip_get_neighbors # MUDAAAAAAAAAARR !!!!!!!!!!!!!
)
from cifo.problem.objective import ProblemObjective
from cifo.algorithm.ga_operators import (
    initialize_using_random,
    roulettewheel_selection, rank_selection, tournament_selection,
    singlepoint_crossover, cycle_crossover, order1_crossover,
    single_point_mutation, swap_mutation,
    elitism_replacement, standard_replacement 
)    
from cifo.util.terminal import Terminal, FontColor
from cifo.util.observer import LocalSearchObserver
from random import randint
import numpy as np

def plot_performance_chart( df ):
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

tsp_decision_variables = {
    "Distances" : np.array(input), #<< Number, Mandatory
    "Item-Name" : [i for i in range(1,len(input[0]))] # << String, Optional, The names of the items, can be used to present the solution.
        # It starts at 1 reserving the 0 for the static city
        # We are setting the departure point, so we only have to arrange the n-1 remaining cities
}

# Problem Instance
tsp_problem_instance = TravelSalesmanProblem(
    decision_variables = tsp_decision_variables
)


# Configuration
#--------------------------------------------------------------------------------------------------
# parent selection object
#parent_selection = TournamentSelection()
parent_selection = roulettewheel_selection

params = {
        # params
        "Population-Size"           : 10,
        "Number-of-Generations"     : 100,
        "Crossover-Probability"     : 0.8,
        "Mutation-Probability"      : 0.8,
        # operators / approaches
        "Initialization-Approach"   : initialize_using_random,
        "Selection-Approach"        : parent_selection,
        "Tournament-Size"           : 5,
        "Crossover-Approach"        : order1_crossover,
        "Mutation-Approach"          : swap_mutation,
        "Replacement-Approach"      : elitism_replacement
    }

log_name = "mp0-8"

number_of_runs = 30

# Run the same configuration many times
#--------------------------------------------------------------------------------------------------
for run in range(1, number_of_runs + 1):
    # Genetic Algorithm
    ga = GeneticAlgorithm( 
        problem_instance = tsp_problem_instance,
        params =  params,
        run = run,
        log_name = log_name
        )

    ga_observer = LocalSearchObserver(ga)
    ga.register_observer(ga_observer)
    ga.search()    
    ga.save_log()

# Consolidate the runs
#--------------------------------------------------------------------------------------------------

# save the config

# consolidate the runs information
from os import listdir, path, mkdir
from os.path import isfile, join
from pandas import pandas as pd
import numpy as np

log_dir   = f"./log/{log_name}" 

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
df["Fitness_Lower"]  = df["Fitness_Mean"] - 1.96*df["Fitness_SD"]/(number_of_runs**0.5)
df["Fitness_Upper"]  = df["Fitness_Mean"] + 1.96*df["Fitness_SD"]/(number_of_runs**0.5)


if not path.exists(log_dir):
    mkdir(log_dir)

df.to_excel(log_dir + "/all.xlsx", index = False, encoding ='utf-8')

plot_performance_chart(df)




#[sum(sublist) for sublist in itertools.izip(*myListOfLists)]



