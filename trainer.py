import torch
from torch import nn
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time
from sklearn.metrics import (confusion_matrix, precision_score,
                             recall_score, f1_score)


class Trainer():

    def __init__(self, net, loc, timeline, optimizer, device,
                 combinations_list, batch_size, trainloader, validationloader,
                 testloader):

        self.device = device
        self.loc = loc
        self.timeline = timeline
        self.combinations_list = combinations_list
        self.trainloader = trainloader
        self.testloader = testloader
        self.validationloader = validationloader
        self.batch_size = batch_size
        self.net = net
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
                        'horse', 'ship', 'truck')
        self.lr_step = 0.001
        self.epoch_stage_lr_steps = 20
        self.scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    def _train_epoch(self, epoch, total_train_accuracy, total_train_loss):
        self.net.train()
        # iterate through the train loader

        running_loss = 0
        correct_train = 0
        total_train = 0
        print("\nEpoch: {} and learning Rate: {}".format(
                                        (epoch), self.scheduler.get_last_lr()))

        time_reading_from_loader = []
        time_transfer_to_device = []
        time_forward_and_backprop = []
        time_loss_acc = []

        start_reading_from_loader = time.time()

        # for i, data in enumerate(self.trainloader, 0):
        for i, (inputs, labels) in enumerate(self.trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data
            # zero the parameter gradients
            # optimizer.zero_grad()
            # for parameter in self.net.parameters():
            #     parameter.grad = None

            end_reading_from_loader = time.time()

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            end_transfer_to_device = time.time()

            self.optimizer.zero_grad(set_to_none=True)

            # forward + backward + optimize
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            end_forward_and_backprop = time.time()

            running_loss += (loss.item())
            _, predicted1 = torch.max(outputs.data, 1)

            for labels, predicted1 in zip(labels, predicted1):
                if labels == predicted1:
                    correct_train += 1
                total_train += 1

            end_loss_pred_data_collection = time.time()

            time_reading_from_loader.append(end_reading_from_loader -
                                            start_reading_from_loader)

            time_transfer_to_device.append(end_transfer_to_device -
                                           end_reading_from_loader)
            time_forward_and_backprop.append(end_forward_and_backprop -
                                             end_transfer_to_device)
            time_loss_acc.append(end_loss_pred_data_collection -
                                 end_forward_and_backprop)

            start_reading_from_loader = time.time()

        total_train_accuracy.append(100*correct_train/total_train)
        total_train_loss.append(running_loss/self.batch_size)

        print("Epoch: {}, training loss: {:.3f} and acc: {:.3f} %\n".format(
            (epoch), running_loss/self.batch_size,
            100*(correct_train/total_train)))

        print("Time for the tasks to complete: \n")

        print("Reading_from_loader_takes :{} seconds".format(
                np.mean(time_reading_from_loader)))

        print("Transfer_to_device_takes :{} seconds".format(
                np.mean(time_transfer_to_device)))

        print("Forward_and_backprop :{} seconds".format(
                np.mean(time_forward_and_backprop)))

        print("Loss_and_Acc_parameters :{} seconds\n".format(
                np.mean(time_loss_acc)))

        return total_train_accuracy, total_train_loss

    def _val_epoch(self, cut_off, epoch, total_epoch, total_validation_loss,
                   total_validation_accuracy, val_acc_list, early_stop):
        self.net.eval()

        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():

            for info in self.validationloader:
                inputs1, labels1 = info
                inputs1 = inputs1.to(self.device)
                labels1 = labels1.to(self.device)
                outputs1 = self.net(inputs1)
                new_loss = self.criterion(outputs1, labels1)
                val_loss += (new_loss.item())
                _, predicted2 = torch.max(outputs1.data, 1)

                for labels1, predicted2 in zip(labels1, predicted2):
                    total_val += 1
                    if labels1 == predicted2:
                        correct_val += 1

        total_validation_accuracy.append(100*correct_val/total_val)
        total_validation_loss.append(val_loss/self.batch_size)
        total_epoch.append(epoch)

        print("Epoch: {}, validation loss: {:.3f} and acc: {:.3f} %".format(
            (epoch), val_loss/self.batch_size, 100*(correct_val/total_val)))

        if (epoch > 1):
            if(total_validation_loss[-1] > (total_validation_loss[-2])):
                cut_off += 1

            if cut_off > 2:
                early_stop = True

        else:
            cut_off = 0

        if((epoch % 5 == 0) and (epoch > 4)):
            if((total_validation_loss[-1] / total_validation_loss[-5]) > 0.90):
                early_stop = True

        if early_stop is True:
            os.makedirs(self.loc)
            torch.save(self.net.state_dict(), (self.loc + "/net.pt"))

        self.scheduler.step()

        val_accuracy = 100*(correct_val/total_val)

        return (early_stop, total_epoch, val_accuracy, total_validation_loss,
                total_validation_accuracy, correct_val, total_val)

    def fit(self, val_acc_list, epochs):
        cut_off = 0
        total_epoch = []
        total_train_loss = []
        total_test_loss = []
        total_train_accuracy = []
        total_test_accuracy = []
        early_stop = False

        for epoch in range(epochs):
            total_train_accuracy, total_train_loss = self._train_epoch(
                epoch, total_train_accuracy, total_train_loss)

            (early_stop, total_epoch, val_accuracy, total_validation_loss,
             total_validation_accuracy, correct_val, total_val) = (
             self._val_epoch(cut_off, epoch, total_epoch, total_test_loss,
                             total_test_accuracy, val_acc_list, early_stop))

            if early_stop is True:
                print("Early stop initialized")
                break
        if early_stop is False:
            os.makedirs(self.loc)
            torch.save(self.net.state_dict(), (self.loc + "/net.pt"))

        val_acc_list.append(100*correct_val/total_val)
        print("list of acc : {}\n".format(val_acc_list))

        print('Finished Training')

        return (total_epoch, val_accuracy, total_train_accuracy,
                total_train_loss, total_validation_accuracy,
                total_validation_loss)

    def evaluate(self):
        # evaluation after training is finished

        # iterate through test loader
        self.net.load_state_dict(torch.load(self.loc + "/net.pt"))

        correct_pred = {classname: 0 for classname in self.classes}
        total_pred = {classname: 0 for classname in self.classes}

        # no gradients needed
        with torch.no_grad():
            cm_labels = []
            cm_predictions = []

            for data in self.testloader:
                images, labels = data

                images = images.to(self.device)
                outputs = self.net(images)
                _, predictions = torch.max(outputs, 1)

                cm_labels += labels
                predictions = predictions.to("cpu")
                cm_predictions += predictions

                for label, predictions in zip(labels, predictions):

                    if label == predictions:
                        correct_pred[self.classes[label]] += 1
                    total_pred[self.classes[label]] += 1

            cm = confusion_matrix(cm_labels, cm_predictions)
            print(self.classes)
            print(cm)
            print("\n")

            preci = precision_score(cm_labels, cm_predictions,
                                    average='weighted',
                                    labels=np.unique(cm_predictions))

            rec = recall_score(cm_labels, cm_predictions, average='weighted')

            f1_score_data = f1_score(cm_labels,
                                     cm_predictions,
                                     average='weighted')

            # f1_score = ((2*preci*rec)/(preci + rec))
            print("\n")

            model_perf_indicators = {"precision": preci,
                                     "recall": rec,
                                     "f1_score": f1_score_data}

            print("model_perf_indicators : \n", model_perf_indicators)
            print("\n")

            # print accuracy for each class
            for classname, correct_count in correct_pred.items():
                accuracy = 100 * float(correct_count) / total_pred[classname]
                print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                                     accuracy))
            print("\n")

            return cm, model_perf_indicators

    def create_plots(self, cm, total_epoch, total_train_accuracy,
                     total_train_loss, total_validation_accuracy,
                     total_validation_loss):
        # Loss
        plt.figure(figsize=(10, 8))
        plt.plot(total_epoch, total_train_loss)
        plt.plot(total_epoch, total_validation_loss)
        plt.legend(["Train loss", "Validation loss"])
        plt.savefig(self.loc + "/loss.jpg")
        plt.show()

        # Accuracy
        plt.plot(total_epoch, total_train_accuracy)
        plt.plot(total_epoch, total_validation_accuracy)
        plt.legend(["Training Accuracy", "Validation Accuracy"])
        plt.savefig(self.loc + "/Accuracy.jpg")
        plt.show()

        # confusion matrix plot
        plt.figure(figsize=(15, 15))
        plt.matshow(cm, fignum=1)
        plt.colorbar()
        plt.title('Confusion matrix', fontsize=32)
        plt.xlabel(("Actual label"), fontsize=21)
        plt.ylabel(("predicted label"), fontsize=21)
        plt.yticks((range(0, 10)), self.classes, fontsize=15)
        plt.xticks((range(0, 10)), self.classes, fontsize=15)
        for (i, j), z in np.ndenumerate(cm):
            plt.text(j, i, '{:0.1f}'.format(z), ha="center", v="center")
        plt.savefig(self.loc + "/ConfusionMatrix.jpg")
        plt.show()

    def save(self, model_perf_indicators, total_train_loss,
             total_validation_loss, total_train_accuracy,
             total_validation_accuracy, lr_step, epoch_stage_lr_steps):

        # save all the training metrics
        with open(self.loc + "/data parameters.txt", 'w') as w:
            for key in self.combinations_list.keys():
                w.write("{} : {} \n".format(key, self.combinations_list[key]))
            w.write('\n')
            for key in model_perf_indicators.keys():
                w.write("{} : {} \n".format(key, model_perf_indicators[key]))
            w.write("\nEpoch_stage_lr_steps: {} \n".format(
                                             str(self.epoch_stage_lr_steps)))
            w.write("Lr_step: {} \n".format(str(self.lr_step)))

        datalist = [total_train_loss, total_validation_loss,
                    total_train_accuracy, total_validation_accuracy]

        names = ["train_loss.pt", "validation_loss.pt",
                 "train_acc.pt", "validation_acc.pt"]

        for name, data in zip(names, datalist):
            torch.save(data, os.path.join(self.loc, str(name)))
