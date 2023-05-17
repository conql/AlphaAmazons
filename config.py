import neptune

use_gpu = True
lr = 0.001
batch_size = 64
epochs = 2


def init():
    global run
    run = neptune.init_run(project="conql/Amazons")
    run["use_gpu"] = use_gpu
    run["lr"] = lr
    run["batch_size"] = batch_size
    run["epochs"] = epochs
    
