import neptune

use_gpu = True
lr = 0.001
batch_size = 64
epochs = 1


def init():
    global run
    run = neptune.init_run(project="conql/Amazons", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZDMxZjA4Ny1lOTYzLTQ2YjUtYTg1My1iMjQxNDQ3ZTZkMDEifQ==")
    run["use_gpu"] = use_gpu
    run["lr"] = lr
    run["batch_size"] = batch_size
    run["epochs"] = epochs
    
