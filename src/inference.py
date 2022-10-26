"""Module for torch model inference"""

import logging

import numpy as np
import torch

from src.model import Net


def model_inference(
    model: Net,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
    test_inference: bool = False,
) -> set:
    """Make prediction on sample (dataloader)

    Args:
        model (torch.nn.modules.module._IncompatibleKeys): pytorch NN model
        dataloader (DataLoader): torch dataloader
        device (str): device for computing results
        test_inference(bool): flag that use one iteration for model inference

    Return:
        set: Set with label and predictions on data from dataloader
    """

    labels_list = np.array([])
    predictions_list = np.array([])

    number_iterations = len(dataloader.dataset) // dataloader.batch_size + 1

    model.to(device)
    model.eval()
    with torch.no_grad():
        logging.info(
            f"Model sent to {device}, set to eval mode, gradient calculation disabled"
        )
        logging.info("Start model inference process")
        for i, data in enumerate(dataloader):
            if i % 10 == 0:
                logging.info("Start iteration: %d / %d" % i, number_iterations)
            images, label = data[0].to(device), data[1].cpu().detach().numpy()

            predictions = model.forward(images)

            predictions = predictions.squeeze().cpu().detach().numpy()

            labels_list = np.append(labels_list, label)
            predictions_list = np.append(predictions_list, predictions)

            if test_inference:
                logging.info(f"Test inference is {test_inference}")
                break

    return labels_list, predictions_list
