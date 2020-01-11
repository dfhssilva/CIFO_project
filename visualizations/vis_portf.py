import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cifo.custom_problem.portfolio_investment_problem import get_dv_pip

indexes = np.array([407, 33, 65, 290, 342, 208, 351, 27, 271, 99, 268, 395, 400, 369, 250, 349, 374, 389, 466, 108, 341, 444,
           440, 436, 177, 305, 354, 255, 449, 223])

run_dir = r"D:\Google Drive\CIFO Project\PIP Runs\I-rand_S-rol_C-cross_M-insert_R-std_CP-0.1_MP-0.1_PS-40_TS-5_G-1000.xlsx"

# inputs
closing_prices = pd.read_excel(r".\data\sp_12_weeks.xlsx")  # importing data example
exp, cov = get_dv_pip(closing_prices)
var = np.diag(cov)

# representation
config = pd.read_excel(run_dir, sheet_name="Overall_Best_Solution")
representation = np.array(list(map(int, re.findall(r"\d+", config.loc[0, "Representation"]))))

# portfolio and risk free asset
data = np.array([-1, config.loc[0, "Expected Return"], config.loc[0, "Risk"],
                 -2, config.loc[0, "Above Risk Free"], 0]).reshape(2, 3)

# assets in portfolio, expected return, variance and proportion of portfolio
assets = indexes[representation != 0]
# prop = representation[representation != 0]/representation.sum()
data = np.concatenate((data,
                       np.reshape(np.concatenate((assets, exp[assets], var[assets])), (assets.shape[0], 3), order="F")))
plot_df = pd.DataFrame(data, columns=["Id", "Expected", "Variance"])

plt.plot(plot_df["Variance"], plot_df["Expected"], 'bo')







