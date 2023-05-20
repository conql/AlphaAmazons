import neptune

use_gpu = True
lr = 0.001
batch_size = 256
epochs = 10
residual_blocks = 20
residual_channels = 64
model_name = "resnet_02"


def init():
    global run
    run = neptune.init_run(
        project="conql/Amazons",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZDMxZjA4Ny1lOTYzLTQ2YjUtYTg1My1iMjQxNDQ3ZTZkMDEifQ==",
        capture_stdout=True,
        capture_stderr=True,
        capture_hardware_metrics=True,
        capture_hardware_gpu_metrics=True,
    )
    run["use_gpu"] = use_gpu
    run["lr"] = lr
    run["batch_size"] = batch_size
    run["epochs"] = epochs
    run["residual_blocks"] = residual_blocks
    run["residual_channels"] = residual_channels
    run["model_name"] = model_name
