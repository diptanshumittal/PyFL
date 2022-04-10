from torch import nn
import torch.nn.functional as F
from helper.pytorch.optimizers import SFW
from helper.pytorch.constraints import *
import model.pytorch.resnet as resnet
import model.pytorch.googlenet as googlenet
import torchvision


def create_seed_model(config):
    print(config)
    if config["model"]["model_type"] == "mnist":
        model = Net()
    elif config["model"]["model_type"] == "mnist":
        model = Net()
    elif config["model"]["model_type"] == "resnet20":
        model = resnet.resnet20()
    elif config["model"]["model_type"] == "resnet1202":
        model = resnet.resnet1202()
    elif config["model"]["model_type"] == "resnet110":
        model = resnet.resnet110()
    elif config["model"]["model_type"] == "googlenet":
        model = googlenet.googlenet()
    elif config["model"]["model_type"] == "resnet50":
        model = torchvision.models.resnet50()
    else:
        model = Net()
    if config["loss"] == "neg_log_likelihood":
        loss = nn.NLLLoss()
    elif config["loss"] == "cross_entropy":
        loss = nn.CrossEntropyLoss()
    else:
        loss = nn.NLLLoss()
    if config["optimizer"]["optimizer"] == "SFW":
        cf = config["optimizer"]
        constraints = create_lp_constraints(model, ord=float(cf["ord"]), value=int(cf["value"]),
                                            mode=cf["mode"])
        optimizer = SFW(model.parameters(), constraints=constraints, learning_rate=float(cf["learning_rate"]),
                        momentum=float(cf["momentum"]), rescale=cf["rescale"],
                        weight_decay=float(cf["weight_decay"]))
        make_feasible(model, constraints)
    elif config["optimizer"]["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    elif config["optimizer"]["optimizer"] == "SGD":
        cf = config["optimizer"]
        optimizer = torch.optim.SGD(model.parameters(), lr=float(cf["learning_rate"]),
                                    momentum=float(cf["momentum"]), weight_decay=float(cf["weight_decay"]))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if config["lr_scheduler"]["type"] == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         list(map(int,
                                                                  config["lr_scheduler"]["lrmilestone"].split(" "))),
                                                         gamma=float(config["lr_scheduler"]["gamma"]), last_epoch=-1)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=1, last_epoch=-1)
    return model, loss, optimizer, scheduler


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

