import optuna


# Library for grid search
def lib_grid():
    print("\nLibrary Grid search activated")
    search_space = {"optimizer": ["SGD", "Adam"],
                    "Batch_size": [32, 64, 128],
                    "Learning_rate": [1e-2, 1e-3],
                    "Momentum": [0.3, 0.6, 0.9]}
    study = optuna.create_study(
                            direction="maximize",
                            sampler=optuna.samplers.GridSampler(search_space))
    return study
