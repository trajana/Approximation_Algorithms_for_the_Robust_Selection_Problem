import pickle
import pandas as pd

with open("/home/trajana/Documents/Publikation_Masterarbeit/Manuscript/Code/results/n_2026-03-02_16-23-23/solve_two_branches_smallest_wi/all_results.pkl", "rb") as f:
    data = pickle.load(f)

df = pd.DataFrame(data)

df.to_csv("/home/trajana/Documents/Publikation_Masterarbeit/Manuscript/Code/results/n_2026-03-02_16-23-23/solve_two_branches_smallest_wi/ergebnisse.csv", index=False)