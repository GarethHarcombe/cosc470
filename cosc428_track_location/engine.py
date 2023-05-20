import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    """ Trains model for one epoch"""
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


@torch.inference_mode()
def evaluate(model, data_loader, device):
    """Evaluate the model by running test images through it and calculating RMSE
    Returns the worst performing image for debugging """
    n_threads = torch.get_num_threads()

    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    squared_error = 0

    worst_image_error = None
    worst_image = None
    worst_preds = None

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        try:
            pred_keys = outputs[0]["keypoints"][0]
            test_keys = targets[0]["keypoints"][0]

            error = ((pred_keys[0][0] - test_keys[0][0]) ** 2 + (pred_keys[1][0] - test_keys[1][0]) ** 2 +   # x differences
                    (pred_keys[0][1] - test_keys[0][1]) ** 2 + (pred_keys[1][1] - test_keys[1][1]) ** 2)    # y differences

            if worst_image_error is None or error > worst_image_error:
                worst_image_error = error
                worst_image = images
                worst_preds = outputs

            squared_error += error
        except:
            squared_error += 100
           
        evaluator_time = time.time()

        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("RMSE:", (squared_error / len(data_loader)) ** 0.5)
    print("Averaged stats:", metric_logger)

    torch.set_num_threads(n_threads)
    return worst_image[0], worst_preds[0]
