import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_data(run_dir):
    excel_file_list = [f for f in os.listdir(run_dir) if f.endswith('.xlsx')]
    config_list = list(map(lambda x: x[:-5], excel_file_list))

    # Getting configuration dfs
    all_dfs = [pd.read_excel(str(run_dir + excel_file)) for excel_file in excel_file_list]

    # Concatenating and selection columns
    df_all = pd.concat(all_dfs, axis=1)
    drop_cols = [col for col in df_all.columns if ((col[:3] == "run") | (col == "Generation"))]
    df_all.drop(drop_cols, axis=1, inplace=True)

    # Creating MultiIndex for columns (config, stats)
    col_index_1 = config_list
    col_index_2 = ["Fitness_SD", "Fitness_Mean", "Fitness_Lower", "Fitness_Upper"]
    col_mindex = pd.MultiIndex.from_product([col_index_1, col_index_2], names=["config", "stats"])
    df_all = pd.DataFrame(df_all.values, columns=col_mindex)
    return df_all


def plot_configs(df_plot, desired_configs, interest=None):

    def replace_labels(x):
        x = x.replace("-", ": ")
        x = x.replace("_", " | ")
        for i in label_dict:
            x = x.replace(i, label_dict.get(i))
        return x

    # Plot configurations
    plot_info = df_plot.loc[:, (desired_configs, ["Fitness_Mean", "Fitness_Lower", "Fitness_Upper"])]

    # Figure
    fig, ax = plt.subplots()
    for i in desired_configs:  # Plot Average and CI
        ax.plot(np.arange(0, 1001), plot_info.loc[:, (i, "Fitness_Mean")])
        ax.fill_between(np.arange(0, 1001), plot_info.loc[:, (i, "Fitness_Lower")],
                        plot_info.loc[:, (i, "Fitness_Upper")], alpha=.3)
    # Set axes labels
    ax.set_xlabel("Generations")
    ax.set_ylabel("Fitness")
    ax.set_xlim(xmin=0, xmax=1000)
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Set legend
    label_dict = dict(zip(["rand", "hc", "greedyI", "mixIB", "mixI", "rol", "tourn", "rank", "cycle", "pmx", "order1",
                           "heur", "mixCB", "mixC", "swap", "insert", "invert", "scramble", "greedyM", "mixMB", "mixM",
                           "elit", "std"],
                          ["Random", "Hill Climbing", "Greedy", "Mix2", "Mix1", "Roulette", "Tournament", "Rank",
                           "Cycle",
                           "PMX", "Order1", "Heuristic", "Mix2", "Mix1", "Swap", "Insert", "Invert", "Scramble",
                           "Greedy", "Mix2", "Mix1", "Elitism", "Standard"]))
    if interest is None:
        ax.legend(list(map(replace_labels, desired_configs)), title="Configurations", bbox_to_anchor=(1, 1.15))
    elif interest == "Mutation":
        ax.legend(list(map(replace_labels, desired_configs)), title="Configurations", bbox_to_anchor=(1, 1.15))
    plt.show()


# -------------------------------------------------- RUN SECTION -------------------------------------------------------
# Getting file names and config names
run_dir = "./visualizations/runs/"
# run_dir = r"D:/Documents/University/7_semestre/CIFO/CIFO_project/log_all/"
df_all = get_data(run_dir)

# Plotting the configs
desired_configs = ["I-mixIB_S-tourn_C-heur_M-mixMB_R-elit_CP-0.1_MP-0.1_PS-20_TS-10_G-1000",
                   "I-mixIB_S-tourn_C-heur_M-invert_R-elit_CP-0.1_MP-0.9_PS-20_TS-15_G-1000",
                   "I-mixIB_S-tourn_C-heur_M-mixMB_R-elit_CP-0.1_MP-0.9_PS-20_TS-10_G-1000",
                   "I-mixIB_S-tourn_C-heur_M-mixMB_R-elit_CP-0.1_MP-0.9_PS-20_TS-15_G-1000"]
plot_configs(df_all, desired_configs)
