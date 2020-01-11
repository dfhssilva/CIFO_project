import pandas as pd
import matplotlib.pyplot as plt
import re

#  import cities and coordinates
cities = pd.read_excel("D:/Google Drive/CIFO Project/TST_data.xlsx", sheet_name="GEN DATA", usecols="A:C")

# import config for tour
config_path = "./log_all/I-mixIB_S-tourn_C-heur_M-mixMB_R-elit_CP-0.1_MP-0.9_PS-20_TS-10_G-1000.xlsx"
# config_path = "D:/Google Drive/CIFO Project/Final Runs/I-mixIB_S-tourn_C-heur_M-mixMB_R-elit_CP-0.1_MP-0.9_PS-20_TS-10_G-1000.xlsx"
# config_path = "D:/Google Drive/CIFO Project/Runs/I-hc_S-tourn_C-heur_M-invert_R-elit_CP-0.1_MP-0.9_PS-20_TS-10_G-1000.xlsx"
config = pd.read_excel(config_path, sheet_name="Overall_Best_Solution")

# sort cities
order = list(map(int, ['0'] + re.findall(r'\d+', config.loc[0, "Representation"])))
cities["City_Ord"] = pd.Categorical(cities["City"], categories=order, ordered=True)
cities.sort_values('City_Ord', inplace=True)
cities.reset_index(inplace=True, drop=True)

# add origin city to bottom of cities

coord_orig = cities.loc[0, ["X", "Y"]].values.tolist()
coord_first = cities.loc[1, ["X", "Y"]].values.tolist()
plot_df = cities.append(cities.loc[cities["City"] == 0])

# plot tour
fig, ax = plt.subplots()
ax.plot(plot_df["X"], plot_df["Y"], marker="o")  # tour
ax.plot(coord_orig[0], coord_orig[1], marker="o", color="r")  # starting city
ax.plot(coord_first[0], coord_first[1], marker="o", color="y")  # first city
plt.show()
