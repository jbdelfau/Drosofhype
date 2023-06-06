import xgboost as xgb
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from lib import Drosophype


def evaluate_performances(preds, real_labels):
    total_abs_diff = map(lambda x: sum([abs(y) for y in x]), preds - real_labels)
    return np.mean(list(total_abs_diff))


model = xgb.XGBRegressor
config = {"population_size": 4,
          "n_parents": 2,
          "mutation_rate": 0.3,
          "additional_properties": [("date_test", datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))],
          "hyperparameters": {
              "n_jobs": {"type": "constant", "value": -1},
              "learning_rate": {"min": 0.01, "max": 1.0, "range": (-0.5, 0.5), "type": "float", "round": 2},
              "n_estimators": {"min": 10, "max": 700, "range": (-100, 100), "type": "int"},
              "max_depth": {"min": 1, "max": 15, "range": (-5, 5), "type": "int"},
              "min_child_weight": {"min": 0, "max": 10, "range": (-5, 5), "type": "int"},
              "gamma": {"min": 0.01, "max": 10.0, "range": (-2, 2), "type": "float", "round": 2},
              "subsample": {"min": 0.01, "max": 1.0, "range": (-0.5, 0.5), "type": "float", "round": 2},
              "colsample_bytree": {"min": 0.01, "max": 1.0, "range": (-0.5, 0.5), "type": "float", "round": 2}
          }
          }

data = pickle.load(open("data/xgbregressor/test_data.pkl", 'rb'))

gen_search = Drosophype(model, config, multioutput=True, multioutput_type="RegressorChain")

# Evaluation function
gen_search.evaluate_performances = evaluate_performances

# Stopping criteria
windows = 10
progress_threshold = 1
max_generations = 15

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
all_results.to_csv("data/xgbregressor/results_test.csv")
