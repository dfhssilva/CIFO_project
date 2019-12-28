import pandas as pd
import os
run_dir = "D:/Google Drive/CIFO Project/Runs/"
excel_file_list = [f for f in os.listdir(run_dir) if f.endswith('.xlsx')]

all_data = pd.DataFrame()
for excel_file in excel_file_list:

        df = pd.read_excel(str(run_dir+excel_file))  # Read the excel file to data frame
        all_data = pd.concat([all_data, df], axis=1)

test = all_data.loc[:, ["Fitness_Mean", "Fitness_Mean", "Fitness_Lower", "Fitness_Upper"]]
