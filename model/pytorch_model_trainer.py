from copy import deepcopy
import os
import sys

sys.path.append(os.getcwd())
import yaml
import torch
import threading
import collections
import time
from torch.utils.data import DataLoader
from helper.pytorch.pytorch_helper import PytorchHelper
from model.pytorch.pytorch_models import create_seed_model

def weights_to_np(weights):
    weights_np = collections.OrderedDict()
    for w in weights:
        weights_np[w] = weights[w].cpu().detach().numpy().tolist()
    return weights_np


def np_to_weights(weights_np):
    weights = collections.OrderedDict()
    for w in weights_np:
        weights[w] = torch.tensor(weights_np[w])
    return weights


def grad_to_np(grad_tensor):
    grad_np = []
    for i in range(len(grad_tensor)):
        temp = collections.OrderedDict()
        for w in grad_tensor[i]:
            temp[w] = grad_tensor[i][w].cpu().detach().numpy().tolist()
        grad_np.append(temp)
    return grad_np


def np_to_grad(grad_np, device="cpu"):
    grad_tensor = []
    for i in range(len(grad_np)):
        temp = collections.OrderedDict()
        for w in grad_np[i]:
            temp[w] = torch.tensor(grad_np[i][w]).to(device)
        grad_tensor.append(temp)
    return grad_tensor


class PytorchModelTrainer:
    def __init__(self, config):
        self.global_momentum = None
        self.helper = PytorchHelper()
        self.stop_event = threading.Event()
        self.global_model_path = config["global_model_path"]
        self.model, self.loss, self.optimizer, self.scheduler = create_seed_model(config)
        self.device = torch.device(config["cuda_device"])
        self.loss = self.loss.to(self.device)
        self.model.to(self.device)
        self.round_type = "fedavg"  # config["round"]["type"]
        self.config = config
        self.momentum_change = None
        print("Device being used for training :", self.device, flush=True)
        print("Loading dataset")
        self.train_loader = DataLoader(
            self.helper.read_data(config["data"]["dataset"], config["data_path"], config["train_samples"], True),
            batch_size=int(config["data"]['batch_size']), shuffle=True, pin_memory=True)
        self.test_loader = DataLoader(
            self.helper.read_data(config["data"]["dataset"], config["data_path"], config["test_samples"], False),
            batch_size=int(config["data"]['batch_size']), shuffle=True, pin_memory=True)
        print("Dataset loaded")

    def evaluate(self, dataloader):
        self.model.eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            for x, y in dataloader:
                if self.stop_event.is_set():
                    raise ValueError("Round stop requested by the reducer!!!")
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss += self.loss(output, y).item() * x.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()
            loss /= len(dataloader.dataset)
            acc = correct / len(dataloader.dataset)
        return float(loss), float(acc)

    def validate(self):
        print("-- RUNNING VALIDATION --", flush=True)
        try:
            training_loss, training_acc = self.evaluate(self.train_loader)
            test_loss, test_acc = self.evaluate(self.test_loader)
        except Exception as e:
            print("failed to validate the model {}".format(e), flush=True)
            raise
        report = {
            "classification_report": 'evaluated',
            "status": "pass",
            "training_loss": training_loss,
            "training_accuracy": training_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        }
        print("-- VALIDATION COMPLETED --", flush=True)
        return report

    def train(self, settings):
        print("---- RUNNING TRAINING ----", flush=True)
        self.model.train()
        for i in range(settings['epochs']):
            for x, y in self.train_loader:
                if self.stop_event.is_set():
                    raise ValueError("Round stop requested by the reducer!!!")
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.step(self.global_momentum)
                self.optimizer.zero_grad()
                output = self.model(x)
                error = self.loss(output, y)
                error.backward()
                self.global_momentum, self.momentum_change = self.optimizer.update_momentum(self.global_momentum,
                                                                                            self.momentum_change)
            self.scheduler.step()
        print("---- TRAINING COMPLETED ----", flush=True)

    def start_round(self, round_config, stop_event):
        self.stop_event = stop_event
        try:
            try:
                print("Loading global model", flush=True)
                model, momentum, momentum_change = self.helper.load_model(self.global_model_path)
                self.global_momentum = np_to_grad(momentum, self.device)
                self.momentum_change = np_to_grad(momentum_change, self.device)
                self.model.load_state_dict(np_to_weights(model))
                print("Conversion sucess")
                # self.global_model = deepcopy(self.model)
                self.model.to(self.device)
            except Exception as e:
                print("Error while loading global model", e)
                raise ValueError("Not able to load global model")
            print("Loaded global model successfully on", self.device, flush=True)
            self.train(round_config)
            # if self.round_type == "fedavg":
            #     for i in range(round_config['epochs']):
            #         print('Running epoch {} with LR {:.5e}'.format(i, self.optimizer.param_groups[0]['lr']), flush=True)
            #         train(self.train_loader, self.model, self.loss, self.optimizer, i + 1, self)
            #         self.scheduler.step()
            #         if self.stop_event.is_set():
            #             raise Exception("Stop requested")
            # elif self.round_type == "fedprocx":
            #     base_loss, base_acc = self.evaluate(self.train_loader)
            #     i = 1
            #     thresh = self.config["round"]["gamma"] * base_loss
            #     print("Base training loss = ", base_loss, " threshold = ", thresh, flush=True)
            #     while True:
            #         print('Running epoch {} with LR {:.5e}'.format(i, self.optimizer.param_groups[0]['lr']), flush=True)
            #         train(self.train_loader, self.model, self.loss, self.optimizer, i, self)
            #         loss, _ = self.evaluate(self.train_loader)
            #         loss += self.config["round"]["mu"] * sum((y - x).sum() for x, y in
            #                                                  zip(self.global_model.state_dict().values(),
            #                                                      self.model.state_dict().values()))
            #         self.scheduler.step()
            #         print("Loss in the epoch", loss)
            #         if loss < thresh:
            #             break
            #         i += 1
            #         if self.stop_event.is_set():
            #             raise Exception("Stop requested")
            report = self.validate()
            self.model.cpu()
            model = weights_to_np(self.model.state_dict())
            model["momentum"] = grad_to_np(self.global_momentum)
            self.helper.save_model(model, self.global_model_path)
            self.momentum_change = None
            return report
        except Exception as e:
            print("Error while running the round ", e, flush=True)
            return {"status": "fail"}


def train(train_loader, model, criterion, optimizer, epoch, model_trainer):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        if model_trainer.stop_event.is_set():
            raise ValueError("Round stop requested by the reducer!!!")
        data_time.update(time.time() - end)
        target = target.to(model_trainer.device)
        input_var = input.to(model_trainer.device)
        target_var = target
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        # print(model.parameters().grad)
        optimizer.step()
        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # if i % 10 == 0 and i > 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
        #         epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, top1=top1))
    return losses.avg


def validate(val_loader, model, criterion, model_trainer):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.to(model_trainer.device)
            input_var = input.to(model_trainer.device)
            target_var = target.to(model_trainer.device)
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
            output = output.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 50 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    return top1.avg, losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == "__main__":
    with open('settings/settings-common.yaml', 'r') as file:
        try:
            client_config = dict(yaml.safe_load(file))
        except yaml.YAMLError as e:
            print('Failed to read model_config from settings file', flush=True)
            raise e
    print("Setting files loaded successfully !!!")
    client_config["training"]["cuda_device"] = "cuda:0"
    client_config["training"]["directory"] = "data/clients/" + "1" + "/"
    client_config["training"]["data_path"] = client_config["training"]["directory"] + "data.npz"
    client_config["training"]["global_model_path"] = client_config["training"]["directory"] + "weights.npz"
    client_config["training"]["train_samples"] = 1000
    client_config["training"]["test_samples"] = 500
    model_trainer = PytorchModelTrainer(client_config["training"])
    model_trainer.model.to(model_trainer.device)
    stop_round_event = threading.Event()
    model_trainer.stop_event = stop_round_event
    for i in range(2):
        print('current lr {:.5e}'.format(model_trainer.optimizer.param_groups[0]['lr']))
        train(model_trainer.train_loader, model_trainer.model, model_trainer.loss, model_trainer.optimizer, i + 1,
              model_trainer)
        model_trainer.scheduler.step()
        validate(model_trainer.test_loader, model_trainer.model, model_trainer.loss, model_trainer)
