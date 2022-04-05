import torch


def saving(model_perf_indicators, combinations_list, epoch, loc,
           batch_size, optimizer, lr_step, epoch_stage_lr_steps,
           total_train_loss, total_validation_loss, total_train_acc,
           total_validation_acc):

    with open(loc + "/data parameters.txt", 'w') as w:
        for key in combinations_list.keys():
            w.write("{} : {} \n".format(key, combinations_list[key]))
        w.write('\n')
        for key in model_perf_indicators.keys():
            w.write("{} : {} \n".format(key, model_perf_indicators[key]))
        w.write("\nEpoch_stage_lr_steps:{}".format(str(epoch_stage_lr_steps)))
        w.write("\nLr_step: {} \n".format(str(lr_step)))

    datalist = [total_train_loss, total_validation_loss,
                total_train_acc, total_validation_acc]

    names = ["train_loss", "validation_loss", "train_acc", "validation_acc"]

    for i, data in enumerate(datalist):
        torch.save((data), (loc+"/"+str(names[i])+".pt"))
