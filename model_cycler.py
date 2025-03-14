import os
import re
import random
import baseline as bl
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset
from core.TrainingParams import TrainingParams
from core import controller
from core import parallelism
import multiprocessing
import math
import sys


class RandomDataset(Dataset):
  def __init__(self, values, labels):
    super(RandomDataset, self).__init__()
    self.values = values
    self.labels = labels

  def __len__(self):
    return len(self.values)  # number of samples in the dataset

  def __getitem__(self, index):
    return self.values[index], self.labels[index]

#Get the amount of channels required for a model. While this number is usually 3, it allows for dynamic retrieval.
def get_channel_depth(model):
    try:
        model(torch.empty(1, 0, 1, 1))
        search = 0
    except RuntimeError as e:
        search = re.search('(\\d+)[^\\d]+channels', str(e))
    channels = int(search.group(1))
    return channels


def create_random_dataset(model_str, model, samples):
        channels = get_channel_depth(model)
        #This is model dependent, most models have this as a minimum size. Inception v3 has minimum size 229x229
        height = 224
        width = 224
        values = torch.rand(samples, channels, height, width)

        metadata = torchvision.models.get_model_weights(model_str).DEFAULT.meta
        tot_categories = len(metadata["categories"])
        labels = torch.randint(tot_categories,(samples,)).tolist()

        return RandomDataset(values, labels)


def decor_nn(model, dataset, batch_size, progress_file, allocations_file, epochs, desired_deadline, dynamic, distributed = True):
    #NOTE: changes to this must reflect on the baseline objects
    #TODO: make training parameters selectable at runtime
    train_model = torchvision.models.get_model(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(train_model.parameters(), lr=0.001)
    train_object = TrainingParams(train_model, dataset, loss_fn, optimizer, batch_size, math.sqrt)
    alpha = 1
    time = controller.start(train_object, desired_deadline, alpha, epochs, progress_file, allocations_file, dynamic, distributed)

    return time

if __name__ == "__main__":

    samples = 3
    model_list = ["resnet50", "vgg19"]


    deadlines = [1.0, 1.5, 1.8]
    num_batches = 100
    epochs = 2
    batch_size= 32

    tot_data_samples = batch_size * num_batches
    distributed = False
     #This is required to use DDP
    multiprocessing.set_start_method("spawn")
    if len(sys.argv) < 2:
        dynamic = False
    else:
        dynamic = bool(sys.argv[0])

    results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results_gpu")
    if not os.path.exists(results_path):
            os.makedirs(results_path)

    records_file = os.path.join(results_path, "records_batchsize"+str(batch_size)+"_epochs"+str(epochs)+"_numbatches"+str(num_batches)+".csv")

    table_file = os.path.join(results_path, "table_batchsize"+str(batch_size)+"_epochs"+str(epochs)+"_numbatches"+str(num_batches)+".csv")

    with open(records_file, "a+") as file:
        file.write("model,baseline,expected,obtained\n")

    with open(table_file, "a+") as file:
        file.write("model,dlcoefficient,avg_error,earliness,lateness\n")

    for idx, model in enumerate(model_list):

        model_dir = os.path.join(results_path, model)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        for dl_index in range(len(deadlines)):
            max_error = 0
            min_error = 0
            avg_error = 0

            print(f"Testing deadline {deadlines[dl_index]}\n")
            deadline_dir = os.path.join(model_dir, "deadline_" + str(deadlines[dl_index]))
            if not os.path.exists(deadline_dir):
                os.makedirs(deadline_dir)

            for sample in range(samples):
                sample_dir = os.path.join(deadline_dir, "sample_" + str(sample))
                if not os.path.exists(sample_dir):
                    os.makedirs(sample_dir)

                # NOTE: changes to this must reflect on the decor_nn objects
                # TODO: make training parameters selectable at runtime
                train_model = torchvision.models.get_model(model)
                dataset = create_random_dataset(model, train_model, tot_data_samples)
                loss_fn = nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(train_model.parameters(), lr=0.001)
                if distributed:
                    baseline = bl.ddp_run(train_model, loss_fn, optimizer, dataset, epochs, batch_size)
                else:
                    baseline = bl.dp_run(train_model, loss_fn, optimizer, dataset, epochs, batch_size)

                print(f"- Sample {str(sample)} -\n")
                progress_file = os.path.join(sample_dir, "progress_timeline.csv")
                allocations_file = os.path.join(sample_dir, "allocations_timeline.csv")
                desired_deadline = baseline * deadlines[dl_index]


                result = decor_nn(model, dataset, batch_size, progress_file, allocations_file, epochs,
                                                    desired_deadline, bool(dynamic), distributed)
                with open(records_file, "a") as file:
                    file.write(model + "," + str(baseline) + "," + str(desired_deadline) + "," + str(result) +"\n")

                error = ((result - desired_deadline) / desired_deadline) * 100
                if (error < min_error):
                    min_error = ((result - desired_deadline) / desired_deadline) * 100
                elif (error > max_error):
                    max_error = ((result - desired_deadline) / desired_deadline ) * 100

                avg_error += abs(error)
            avg_error /= samples
            with open(table_file, "a+") as file:
                file.write(model + "," + str(deadlines[dl_index]) + "," + str(round(avg_error, 2)) + "," + str(
                    round(min_error, 2)) + "," + str(round(max_error, 2)) + "\n")