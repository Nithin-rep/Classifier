import torch
import matplotlib.pyplot as plt
import os
from torch.optim.lr_scheduler import StepLR


def train_val(device, loc, count, timeline, val_acc_list, combinations_list,
              trainloader, validationloader, optimizer, net, criterion,
              classes, batch_size):

    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    total_train_loss = []
    total_validation_loss = []
    total_train_acc = []
    total_validation_acc = []
    epoch = []
    total_epoch = []
    counter = 0
    early_stop = False
    cut_off = 0
    num_of_epochs = 2

    for epoch in range(num_of_epochs):  # loop over the dataset multiple times
        counter += 1
        running_loss = 0.0
        val_loss = 0.0
        correct_train = 0
        correct_val = 0
        total_train = 0
        total_val = 0
        print("\nEpoch: {} and learning Rate: {}".format(
                                             (epoch), scheduler.get_last_lr()))

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            # optimizer.zero_grad()
            for parameter in net.parameters():
                parameter.grad = None

            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward + backward + optimize
            outputs = net.train()(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += (loss.item())
            _, predicted1 = torch.max(outputs.data, 1)

            for labels, predicted1 in zip(labels, predicted1):
                if labels == predicted1:
                    correct_train += 1
                total_train += 1

        with torch.no_grad():

            for info in validationloader:
                inputs1, labels1 = info
                inputs1 = inputs1.to(device)
                labels1 = labels1.to(device)
                outputs1 = net.eval()(inputs1)
                new_loss = criterion(outputs1, labels1)
                val_loss += (new_loss.item())
                _, predicted2 = torch.max(outputs1.data, 1)

                for labels1, predicted2 in zip(labels1, predicted2):
                    total_val += 1
                    if labels1 == predicted2:
                        correct_val += 1

        total_train_loss.append(running_loss/batch_size)
        total_validation_loss.append(val_loss/batch_size)
        total_epoch.append(counter)

        total_train_acc.append(100*correct_train/total_train)
        total_validation_acc.append(100*correct_val/total_val)

        print("Epoch: {}, training loss: {:.3f} and acc: {:.3f} %".format(
            (epoch), running_loss/batch_size, 100*(correct_train/total_train)))

        print("validation loss: {:.3f} and acc: {:.3f} %".format(
            val_loss/batch_size, 100*(correct_val/total_val)))

        if (epoch > 1):
            if(total_validation_loss[-1] > (total_validation_loss[-2])):
                cut_off += 1

            elif(epoch % 5 == 0):
                if((total_validation_loss[-1]/total_validation_loss[-5]) >
                   0.90):
                    early_stop = True

            if cut_off > 2:
                early_stop = True

        else:
            cut_off = 0

        if early_stop is True:
            os.makedirs(loc)
            torch.save(net.state_dict(), (loc + "/net.pt"))
            print("Early stop initialized")
            break

        scheduler.step()
    print('Finished Training')

    if early_stop is False:
        os.makedirs(loc)
        torch.save(net.state_dict(), (loc + "/net.pt"))

    val_acc_list.append(100*correct_val/total_val)
    print("list of acc : {}\n".format(val_acc_list))

    plt.figure(figsize=(10, 8))
    plt.plot(total_epoch, total_train_loss)
    plt.plot(total_epoch, total_validation_loss)
    plt.legend(["Train loss", "Validation loss"])
    plt.savefig(loc + "/loss.jpg")
    plt.show()

    plt.plot(total_epoch, total_train_acc)
    plt.plot(total_epoch, total_validation_acc)
    plt.legend(["Training Accuracy", "Validation Accuracy"])
    plt.savefig(loc + "/Accuracy.jpg")
    plt.show()

    val_accuracy = 100*(correct_val/total_val)

    return (val_accuracy, total_train_loss, total_validation_loss,
            total_train_acc, total_validation_acc, counter)
