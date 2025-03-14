#! /usr/bin/env python3
import os
import re
import random
import torch
import torchvision
import torch.nn as nn
import controller
import multiprocessing as mp
import subprocess
import math
import main
import sys

if __name__ == "__main__":

    samples = 3
    model_list = ["resnet50", "vgg19"]

    image_name = "test"
    deadlines = [1.0, 1.5, 1.8]
    num_batches = 20
    epochs = 5
    batch_size = 32

    tot_data_samples = batch_size * num_batches

    if len(sys.argv) < 2:
        dynamic = False
    else:
        dynamic = bool(sys.argv[0])
    results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results")

    subprocess.run("docker build -t " + image_name + " --cpu-quota=" + str(
        int(100000 * mp.cpu_count())) + " --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) . ", shell=True,
                   check=True)

    if not os.path.exists(results_path):
        os.makedirs(results_path)
    records_file = os.path.join(results_path,
                                "records_batchsize" + str(batch_size) + "_epochs" + str(epochs) + "_numbatches" + str(
                                    num_batches) + ".csv")
    table_file = os.path.join(results_path,
                              "table_batchsize" + str(batch_size) + "_epochs" + str(epochs) + "_numbatches" + str(
                                  num_batches) + ".csv")
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

                baseline = main.cpu_training(model, num_batches, epochs, batch_size)
                print(f"- Sample {str(sample)} -\n")
                progress_file = os.path.join(sample_dir, "progress_timeline.csv")
                allocations_file = os.path.join(sample_dir, "allocations_timeline.csv")
                desired_deadline = baseline * deadlines[dl_index]

                result = controller.start(model, num_batches, batch_size, desired_deadline, 1, epochs, allocations_file,
                                          progress_file, image_name, bool(dynamic))
                with open(records_file, "a") as file:
                    file.write(model + "," + str(baseline) + "," + str(desired_deadline) + "," + str(result) + "\n")

                error = ((result - baseline) / baseline) * 100
                if (error < min_error):
                    min_error = ((result - baseline) / baseline) * 100
                elif (error > max_error):
                    max_error = ((result - baseline) / baseline) * 100

                avg_error += abs(error)
            avg_error /= samples
            with open(table_file, "a+") as file:
                file.write(model + "," + str(deadlines[dl_index]) + "," + str(round(avg_error, 2)) + "," + str(
                    round(min_error, 2)) + "," + str(round(max_error, 2)) + "\n")