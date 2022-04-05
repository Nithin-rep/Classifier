import os
import torch
import torch.nn as nn
from torch import optim

from dataset_classi import class_data
from net_classi import Net
from tuning.hyper_parameters import parameters_tune
from tuning.hyper_parameters import directory
from datetime import datetime
from tuning.grid_search_lib import lib_grid
from tuning.random_search_lib import lib_rand
from trainer import Trainer


def main(loc_main, device, val_acc_list, combinations_list):

    count = len(val_acc_list)

    batch_size = combinations_list["Batch_size"]

    trainloader, testloader, validationloader = class_data(
                                count, batch_size, trainingset, validationset)

    ########################################################################

    net = Net()
    net.to(device)

    ########################################################################

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    if ((combinations_list["optimizer"]) == ("Adam") or (optim.Adam)):
        optimizer = optim.Adam(net.parameters(),
                               lr=combinations_list["Learning_rate"])

    else:
        optimizer = optim.SGD(net.parameters(),
                              lr=combinations_list["Learning_rate"],
                              momentum=combinations_list["Momentum"])

    lr_step = 0.001
    epoch_stage_lr_steps = 20

    ########################################################################

    loc = loc_main + str(count)

    train = Trainer(net, loc, timeline, optimizer, device, combinations_list,
                    batch_size, trainloader, validationloader, testloader)

    (total_epoch, val_accuracy, total_train_accuracy, total_train_loss,
     total_validation_accuracy, total_validation_loss) = train.fit(
     val_acc_list, epochs=25)

    cm, model_perf_indicators = train.evaluate()

    train.create_plots(cm, total_epoch, total_train_accuracy,
                       total_train_loss, total_validation_accuracy,
                       total_validation_loss)

    train.save(model_perf_indicators, total_train_loss,
               total_validation_loss, total_train_accuracy,
               total_validation_accuracy, lr_step, epoch_stage_lr_steps)

    return val_accuracy


if __name__ == "__main__":

    val_acc_list = []
    timeline = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    loc_main, option, search_type = directory(timeline)
    trainingset = []
    validationset = []

    if torch.cuda.is_available():
        device = "cuda:0"
        print("\ngpu available")
        torch.backends.cudnn.benchmark = True

    else:
        device = "cpu"
        print("\nonly cpu available")

    def tunning_combinations(trial):
        combinations_list = {
          "optimizer": trial.suggest_categorical("optimizer", ["SGD", "Adam"]),
          "Batch_size": trial.suggest_int("Batch_size", 32, 128),
          "Learning_rate": trial.suggest_float("Learning_rate", 0.0001, 0.01),
          "Momentum": trial.suggest_float("Momentum", 0.3, 0.9)
        }
        return combinations_list

    if ((option == "1") or (option == "2")):
        combinations_list = parameters_tune(option)
        for i in range(len(combinations_list)):
            print(combinations_list[i])
            main(loc_main, device, val_acc_list, combinations_list[i])


# Library for grid search

    elif(option == "3"):
        study = lib_grid()

        def opt_tunner(trial):

            combinations_list = tunning_combinations(trial)
            print("\n", combinations_list)
            val_accuracy = main(loc_main, device, val_acc_list,
                                combinations_list)
            return val_accuracy

        study.optimize(opt_tunner)

    # library for random search
    elif(option == '4'):
        study = lib_rand()

        def opt_tunner(trial):
            combinations_list = tunning_combinations(trial)
            print("\n", combinations_list)
            val_accuracy = main(loc_main, device, val_acc_list,
                                combinations_list)
            return val_accuracy

        study.optimize(opt_tunner, n_trials=5)

    # saving loc switch as per option opted

    with open(os.path.join("parameters_and_graphs", search_type, timeline,
                           "Best data parameters.txt"), 'w') as w:

        max_value = max(val_acc_list)
        w.write("Best_accuracy: {}  with  trail: {} \n".format(
            str(max_value), str(val_acc_list.index(max_value))))
        w.write("val_acc_of different_trials: {} \n".format(str(val_acc_list)))
