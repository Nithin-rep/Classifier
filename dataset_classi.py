import torch
import torchvision
import torchvision.transforms as transforms


def class_data(count, batch_size, trainingset, validationset):

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))])
    if count == 0:
        command = True

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True,
                                                transform=transform)

        limit = int(0.8*(len(trainset)))

        for i in range(limit):
            trainingset.append(trainset[i])

        for i in range(len(trainset) - limit):
            validationset.append(trainset[limit + i])

    else:
        command = False

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=command,
                                           transform=transform)

    trainloader = torch.utils.data.DataLoader(trainingset,
                                              batch_size=batch_size,
                                              num_workers=0, pin_memory=True)

    validationloader = torch.utils.data.DataLoader(validationset,
                                                   batch_size=batch_size,
                                                   num_workers=0,
                                                   pin_memory=True)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=True, num_workers=1,
                                             pin_memory=True)

    return trainloader, testloader, validationloader
