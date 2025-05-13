import matplotlib.pyplot as plt
import pandas as pd
from typing import List
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

class Utils :

    @staticmethod
    def evaluate_model_and_store(X: pd.DataFrame, y: pd.Series, name: str, pipe: Pipeline, cv = 5, res = []) :
        results = res.copy()
        result = cross_validate(
            pipe,
            X,
            y,
            scoring = "neg_mean_absolute_error",
            cv = cv,
            n_jobs = -1,
            return_train_score = True
        )
        mae_test_score = -result["test_score"]
        mae_train_score = -result["train_score"]
        results.append(
            {
                "preprocessor": name,
                "mae_test_mean": mae_test_score.mean(),
                "mae_test_std": mae_test_score.std(),
                "mae_train_mean": mae_train_score.mean(),
                "mae_train_std": mae_train_score.std()
            }
        )
        return results
    
    @staticmethod
    def visualize_results(results: List[dict], name: str) -> None :

        results_df = (
            pd.DataFrame(results).set_index("preprocessor").sort_values("mae_test_mean")
        )

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize = (12, 8), sharey = True, constrained_layout = True
        )
        xticks = range(len(results_df))
        name_to_color = dict(
            zip((r["preprocessor"] for r in results), ["C0", "C1", "C2", "C3", "C4", "C5"])
        )

        for subset, ax in zip(["test", "train"], [ax1, ax2]):
            mean, std = f"mae_{subset}_mean", f"mae_{subset}_std"
            data = results_df[[mean, std]].sort_values(mean)
            ax.bar(
                x = xticks,
                height = data[mean],
                yerr = data[std],
                width = 0.9,
                color = [name_to_color[name] for name in data.index],
            )
            ax.set(
                title = f"MAE ({subset.title()})",
                xlabel = name,
                xticks = xticks,
                xticklabels = data.index
            )