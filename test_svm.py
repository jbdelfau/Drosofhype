from sklearn import svm
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from lib import Drosophype


def evaluate_performances(preds, real_labels):
    return np.mean(abs(preds - real_labels))


model = svm.SVC
config = {"population_size": 4,
          "n_parents": 2,
          "mutation_rate": 0.3,
          "additional_properties": [("date_test", datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))],
          "hyperparameters": {
              "kernel": {"type": "category", "possible_values": ["linear", "poly"]},
              "degree": {"min": 1, "max": 4, "range": (-1, 1), "type": "int"},
              "C": {"min": 0.2, "max": 5.0, "range": (-0.5, 0.5), "type": "float", "round": 2}
          }
          }

data = pickle.load(open("data/svm/test_data.pkl", 'rb'))

gen_search = Drosophype(model, config, multioutput=False)

# Evaluation function
gen_search.evaluate_performances = evaluate_performances

# Stopping criteria
windows = 15
progress_threshold = 1
max_generations = 30

all_results = pd.DataFrame()
keep_searching = True
gen = 0
while keep_searching:
    results = gen_search.find_best_params(data["train_data"], data["train_labels"], data["test_data"],
                                          data["test_labels"])
    all_results = pd.concat([all_results, results]).reset_index(drop=True)
    gen = all_results["generation"].max()
    # Stopping criteria: progress under 1% of best score in 10 steps
    if gen >= windows:
        best_score = all_results["performances"].min()
        previous_best_score = all_results[all_results["generation"] <= gen - windows]["performances"].min()
        progress = int((previous_best_score - best_score) / previous_best_score * 100)
        if progress < progress_threshold:
            break
    # Other stopping criteria: no more than 30 generations
    if gen >= max_generations:
        break
    if gen % 5 == 0:
        print(f"Generation: {gen}, best score: {all_results['performances'].min()}")

print(f"Total number of generations: {gen}, final best score: {all_results['performances'].min()}")
all_results.to_csv("data/svm/results_test.csv")
