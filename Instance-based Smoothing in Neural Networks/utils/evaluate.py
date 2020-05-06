"""Evaluates the model"""

import argparse
import logging

import numpy as np
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory of params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir containing weights to load")

"""
This function duplicates "evaluate()" but ignores "loss_fn" simply for speedup purpose.
Validation loss during KD mode would display '0' all the time.
One can bring that info back by using the fetched teacher outputs during evaluation (refer to train.py)
"""
def evaluate_kd(model, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """
    # set model to evaluation mode
    model.eval()
    # summary for current eval loop
    summ = []
    # compute metrics over the dataset
    for i, (data_batch, labels_batch) in enumerate(dataloader):
        # move to GPU if available
        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(async=True), labels_batch.cuda(async=True)
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch) 
        # compute model output
        output_batch = model(data_batch)
        #loss = 0.0  #force validation loss to zero to reduce computation time
        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()
        # compute all metrics on this batch
        #metrics['loss']
        summary_batch = {metric: metrics[metric](output_batch, labels_batch) for metric in metrics}
        # summary_batch['loss'] = loss.data[0]
        #summary_batch['loss'] = loss
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean