# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# NOTE:
# This example is meant as an illustration of how to use CNTKs distributed training feature from the python API.
# The training hyper parameters here are not necessarily optimal and for optimal convergence need to be tuned 
# for specific parallelization degrees that you want to run the example with.

import numpy as np
import sys
import os
from cntk import distributed, device
from cntk.cntk_py import DeviceKind_GPU

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", ".."))

from examples.CifarResNet.CifarResNet import create_reader, train_and_evaluate

#
# Paths relative to current python file.
#
abs_path   = os.path.dirname(os.path.abspath(__file__))
cntk_path  = os.path.normpath(os.path.join(abs_path, "..", "..", "..", ".."))
data_path  = os.path.join(cntk_path, "Examples", "Image", "DataSets", "CIFAR-10")
model_path = os.path.join(abs_path, "Models")

def check_environ(distributed_trainer):
    # check if we have multiple-GPU, and fallback to 1 GPU if not
    # because ResNet uses BatchNormalization which has no implementation on CPU yet
    devices = device.all_devices()
    gpu_count = 0
    for dev in devices:
        gpu_count += (1 if dev.type() == DeviceKind_GPU else 0)
    print("Found {} GPUs".format(gpu_count))
    
    if gpu_count == 0:
        print("No GPU found, exiting")
        quit()

    communicator = distributed_trainer.communicator()
    workers = communicator.workers()
    current_worker = communicator.current_worker()
    print("List all distributed workers")
    for wk in workers:
        if current_worker.global_rank == wk.global_rank:
            print("* {} {}".format(wk.global_rank, wk.host_id))
        else:
            print("  {} {}".format(wk.global_rank, wk.host_id))

    if gpu_count == 1 and len(workers) > 1 :
        print("Warning: running distributed training on 1-GPU will be slow")
        device.set_default_device(gpu(0))
        
if __name__ == '__main__':
    # create the distributed trainer
    warm_start_samples = 50000
    num_quantization_bits = 1
    distributed_trainer = distributed.data_parallel_distributed_trainer(
        num_quantization_bits=num_quantization_bits,
        distributed_after=warm_start_samples)
        
    import datetime
    import time
    print("Start running at {}".format(datetime.datetime.now()))
    start_time = time.time()

    check_environ(distributed_trainer)

    # train the model
    reader_train = create_reader(os.path.join(data_path, 'train_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'), True, distributed_trainer.distributed_after)
    # NOTE test data should not be distributed even when running with distributed trainer
    # which means the all test samples would go through all workers
    reader_test  = create_reader(os.path.join(data_path, 'test_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'), False)

    train_and_evaluate(reader_train, reader_test, max_epochs=5,
        distributed_trainer=distributed_trainer)

    # this is needed to avoid MPI hung in process termination indeterminism
    distributed.Communicator.finalize()
    print("Finished at {}, total {} seconds".format(datetime.datetime.now(), time.time() - start_time))
