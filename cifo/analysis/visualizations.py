import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.offline as pyo
import re

# Getting file names and config names
run_dir = "D:/Google Drive/CIFO Project/Runs/"
# run_dir = "./log/"  # test
excel_file_list = [f for f in os.listdir(run_dir) if f.endswith('.xlsx')]
config_list = list(map(lambda x: x[:-5], excel_file_list))
params = ["Initialization", "Selection", "Crossover", "Mutation", "Replacement", "Crossover_Prob", "Mutation_Prob",
          "Population_Size", "Tournament_Size", "Generations"]

# Getting configurations
all_dfs = [pd.read_excel(str(run_dir + excel_file)) for excel_file in excel_file_list]


def get_configs(config):
    configs = pd.DataFrame(np.repeat(np.array(re.findall(r'-([a-zA-Z]*|\d+\.\d*|\d*)_', config) + [1000]).reshape(1, 10),
                                     1001, axis=0), columns=params)
    return configs


config_dfs = [get_configs(config) for config in config_list]
all_configs_dfs = [pd.concat([all_df, config_df], axis=1) for all_df, config_df in zip(all_dfs, config_dfs)]

# Concatenating and selection columns
df_all = pd.concat(all_configs_dfs, axis=1)
drop_cols = [col for col in df_all.columns if ((col[:3] == "run") | (col == "Generation"))]
df_all.drop(drop_cols, axis=1, inplace=True)

# Creating MultiIndex for columns (config, stats)
col_index_1 = config_list
col_index_2 = ["Fitness_SD", "Fitness_Mean", "Fitness_Lower", "Fitness_Upper", "Initialization", "Selection",
               "Crossover", "Mutation", "Replacement", "Crossover_Prob", "Mutation_Prob", "Population_Size",
               "Tournament_Size", "Generations"]
col_mindex = pd.MultiIndex.from_product([col_index_1, col_index_2], names=["config", "stats"])
df_all = pd.DataFrame(df_all.values, columns=col_mindex)


# Visualizing config performance
def one_line(config, label):
    trace = go.Scatter(
        x=config.index.values,
        y=config["Fitness_Mean"],
        mode='lines',
        name=label,
        error_y=dict(
            type="data",
            array=(config["Fitness_Upper"] - config["Fitness_Lower"]) / 2,
            thickness=0.5,
        )
    )
    return trace


data = [one_line(df_all.loc[:, label], label) for label in col_index_1]

layout = go.Layout(
    paper_bgcolor='rgb(255,255,255)',
    plot_bgcolor='rgb(229,229,229)',
    xaxis=dict(
        gridcolor='rgb(255,255,255)',
        range=[1, 1000],
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
pyo.plot(fig)

# Missing interactive filters.
# References: https://plot.ly/python/reference/#scatter-transforms-items-transform,
# https://plot.ly/python/reference/#layout-updatemenus,
# https://plot.ly/python/filter/,
# https://plot.ly/python/dropdowns/#methods,
# https://plot.ly/python/multiple-transforms/