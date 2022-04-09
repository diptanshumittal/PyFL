#!/usr/bin/python
from datetime import datetime
import threading
import time
import yaml
import argparse
import os
import sys
import subprocess
from multiprocessing import Process
from extras.create_mnist_partitions import create_mnist_partitions
from extras.create_cifar10_dataset import create_cifar10_partitions
from extras.create_cifar100_dataset import create_cifar100_partitions


def run_container(cmd):
    subprocess.call(cmd, shell=True)


parser = argparse.ArgumentParser(description='Federated Learning Client')
parser.add_argument('--type', default='batchscript', type=str, help='Type of initialization')
parser.add_argument('--partition_dataset', default=False, type=bool, help='Type of initialization')
parser.add_argument('--clients', default=1, type=int, help='Type of initialization')
args = parser.parse_args()


def start_clients_docker():
    try:
        for i in range(1, 6):
            with open("docker/client-gpu.yaml", 'r') as file:
                config1 = dict(yaml.safe_load(file))
            print(list(config1["services"].keys()))
            config1["services"]["client" + str(i)] = config1["services"].pop(list(config1["services"].keys())[0])
            config1["services"]["client" + str(i)]["ports"][0] = "809" + str(i) + ":809" + str(i)
            config1["services"]["client" + str(i)]["container_name"] = "client" + str(i)
            config1["services"]["client" + str(i)][
                "command"] = "sh -c 'pip install -r requirements.txt && python client.py " + "data/clients/" + str(
                i) + "/settings-client.yaml'"

            with open('docker/client-gpu.yaml', 'w') as f:
                yaml.dump(config1, f)

            # with open("settings/settings-client.yaml", 'r') as file:
            #     config = dict(yaml.safe_load(file))
            # config["client"]["port"] = 8090 + i
            # config["training"]["data_path"] = "data/clients/" + str(i) + "/mnist.npz"
            # config["training"]["global_model_path"] = "data/clients/" + str(i) + "/weights.npz"
            # with open("data/clients/" + str(i) + "/settings-client.yaml", 'w') as f:
            #     yaml.dump(config, f)
            threading.Thread(target=run_container,
                             args=(
                                 "docker-compose -f docker/client-gpu.yaml up >> data/clients/" + str(i) + "/log.txt",),
                             daemon=True).start()
            time.sleep(10)
    except Exception as e:
        print(e)


def start_clients(clients):
    try:
        available_gpus = ["cuda:0", "cuda:2", "cuda:3", "cuda:3", "cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        for i in range(clients):
            Process(target=run_container,
                    args=("python Client/client.py --gpu=" + available_gpus[i],),
                    daemon=True).start()
            time.sleep(3)
    except Exception as e:
        print(e)


def start_batchscript(clients):
    try:
        gpu = "cuda:0"  # + os.environ["CUDA_VISIBLE_DEVICES"]
        # from Client.client import Client
        # client = Client(args)
        # client.run()
        jobs = []
        for i in range(clients):
            p = Process(target=run_container, args=("python Client/client.py --gpu=" + gpu,))
            p.start()
            jobs.append(p)
        for job in jobs:
            print("Waiting for the clients to end")
            job.join()

    except Exception as e:
        print(e)


def start_clients_slurm(clients):
    try:
        for _ in range(clients):
            Process(target=run_container, args=("sbatch batchscripts/start_client_1.sh",), daemon=True).start()
            time.sleep(1)
    except Exception as e:
        print(e)


def create_dataset_partitions(common_config, no_of_clients):
    if common_config["training"]["dataset"] == "mnist":
        create_mnist_partitions(no_of_clients)
    elif common_config["training"]["dataset"] == "cifar10":
        create_cifar10_partitions(no_of_clients)
    elif common_config["training"]["dataset"] == "cifar100":
        create_cifar100_partitions(no_of_clients)


if __name__ == '__main__':
    if args.partition_dataset:
        with open('settings/settings-common.yaml', 'r') as file:
            try:
                common_config = dict(yaml.safe_load(file))
            except yaml.YAMLError as error:
                print('Failed to read model_config from settings file', flush=True)
                raise error
        create_dataset_partitions(common_config, args.clients)
    if args.type == "batchscript":
        print("Starting the client with SLURM_PROCID = ", os.environ['SLURM_PROCID'])
        start_batchscript(args.clients)
    elif args.type == "slurm":
        start_clients_slurm(args.clients)
    else:
        start_clients(args.clients)
