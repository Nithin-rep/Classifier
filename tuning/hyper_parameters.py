from torch import optim
import random
import os


def directory(timeline):

    print("\nHyperparamter tune option:\n\n")
    print("1) For Manual Grid search, Enter: '1'\n")
    print("2) For Manual Random search, Enter: '2'\n")
    print("3) For Library Grid search, Enter: '3'\n")
    print("4) For Library Random search, Enter: '4'\n")

    option = input("Enter your preference:")

    if option == '1':
        search_type = "grid_search"

    elif option == '2':
        search_type = "random_search"

    elif option == '3':
        search_type = "library_grid_search"

    elif option == '4':
        search_type = "library_random_search"

    loc_main = os.path.join("parameters_and_graphs", search_type,
                            timeline, "trial_")

    return loc_main, option, search_type


def parameters_tune(option):

    dic = {"optimizer": (optim.SGD, optim.Adam),
           "batch_list": (32, 64, 128, 256),
           "lr_list": (0.01, 0.001),
           "momentum_list": (0.3, 0.6, 0.9)}
    combinations_list = []

    # Manual grid search

    if(option == '1'):
        print("\nGrid search activated\n")

        for i in dic["optimizer"]:
            for j in dic["batch_list"]:
                for k in dic["lr_list"]:
                    if i == optim.Adam:
                        combination = {"optimizer": i,
                                       "Batch_size": j,
                                       "Learning_rate": k}
                        combinations_list.append(combination)
                    else:
                        for l in dic["momentum_list"]:
                            combination = {"optimizer": i,
                                           "Batch_size": j,
                                           "Learning_rate": k,
                                           "Momentum": l}
                            combinations_list.append(combination)

    # Manual Random Search

    elif(option == '2'):
        print("\nRandom search ativated\n")
        number_of_random_comb = 30

        for i in range(number_of_random_comb):

            opt = random.sample(dic['optimizer'], k=1)
            batch = random.choice(dic["batch_list"])
            lr = random.choice(dic['lr_list'])
            momentum = random.choice(dic['momentum_list'])

            if (opt == [dic["optimizer"][1]]):
                combination = {"optimizer": optim.Adam,
                               "Batch_size": batch,
                               "Learning_rate": lr}

            else:
                combination = {"optimizer": optim.SGD,
                               "Batch_size": batch,
                               "Learning_rate": lr,
                               "Momentum": momentum}
            combinations_list.append(combination)

    return combinations_list
