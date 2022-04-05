import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt


def test(device, loc, testloader, classes, net):

    net.load_state_dict(torch.load(loc + "/net.pt"))

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # no gradients needed
    with torch.no_grad():
        cm_labels = []
        cm_predictions = []

        for data in testloader:
            images, labels = data

            images = images.to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)

            cm_labels += labels
            predictions = predictions.to("cpu")
            cm_predictions += predictions

            for label, predictions in zip(labels, predictions):

                if label == predictions:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

        cm = confusion_matrix(cm_labels, cm_predictions)
        print(classes)
        print(cm)
        print("\n")

        preci = precision_score(cm_labels, cm_predictions, average='weighted',
                                labels=np.unique(cm_predictions))

        rec = recall_score(cm_labels, cm_predictions, average='weighted')
        f1_score = ((2*preci*rec)/(preci + rec))
        print("\n")

        model_perf_indicators = {"precision": preci,
                                 "recall": rec,
                                 "f1_score": f1_score}

        print("model_perf_indicators : \n", model_perf_indicators)
        print("\n")

        # confusion matrix plot
        plt.figure(figsize=(15, 15))
        plt.matshow(cm, fignum=1)
        plt.colorbar()
        plt.title('Confusion matrix', fontsize=32)
        plt.xlabel(("Actual label"), fontsize=21)
        plt.ylabel(("predicted label"), fontsize=21)
        plt.yticks((range(0, 10)), classes, fontsize=15)
        plt.xticks((range(0, 10)), classes, fontsize=15)
        for (i, j), z in np.ndenumerate(cm):
            plt.text(j, i, '{:0.1f}'.format(z), ha="center", va="center")
        plt.savefig(loc + "/ConfusionMatrix.jpg")
        plt.show()

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                             accuracy))
    print("\n")

    return model_perf_indicators
