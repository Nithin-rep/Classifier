import optuna


def lib_rand():
    print("\n Library Random search activated")
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.RandomSampler())
    return study
