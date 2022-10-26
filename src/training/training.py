"""Module of training torch model for TSP"""

import logging
from typing import List

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class TrainingModel:
    """
    Class for torch model training
    """

    def __init__(
            self,
            model: torch.nn.modules.module._IncompatibleKeys,
            loss: torch.nn.modules.loss,
            optimizer: torch.optim,
            train_dataloader: torch.utils.data.DataLoader,
            val_dataloader: torch.utils.data.DataLoader = None,
            metrics: List = None,
            device: str = 'cuda',
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.metrics = metrics

    @static
    def __train_epoch(self):
        """Private method for train one epoch of model
        Return:
           torch.nn.modules.module._IncompatibleKeys: model after one epoch training
        """

        for i, data in enumerate(train_dataloader):
            if i % 10 == 0:  # every 10 iteration make logging
                logging.info("Start iteration: %d / %d" % i, number_iterations)

            images, labels = data[0].to(device), data[1].to(torch.float32).to(device)

            predictions = self.model.forward(images)  # make predictions
            self.loss = self.loss(torch.squeeze(predictions), labels)  # count loss
            self.loss.backward()  # backpropagation
            self.optimizer.step()  # update weights
            self.optimizer.zero_grad()  # resetting the loss

        return None

