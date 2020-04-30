from argparse import ArgumentParser
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch
import sklearn.metrics as metrics
import numpy as np
# Local imports
from arguments import get_parser
from models import Classifier 
from dataloader import make_loaders

def train(model, dataloader):
    model.network.train()
    for input_batch, target_batch in dataloader:
        model.counter['batches'] += 1
        loss = model.optimize_parameters(input_batch, target_batch)
        mean_loss = get_value(loss)
        model.writer.add_scalar("Training_loss", mean_loss, model.counter['batches'])
    model.counter['epochs'] += 1

def val(model, dataloader):
    model.network.eval()
    accuracy = []
    loss = []
    for input_batch, target_batch in dataloader:
        target_batch = target_batch.to(model.device)
        output, preds = model.predict(input_batch)
        loss.append(model.criterion(output, target_batch).detach().cpu().numpy())
        accuracy.append(metrics.accuracy_score(y_true=np.array(target_batch.detach().cpu().numpy()), y_pred=preds))
    accuracy = np.mean(accuracy)
    loss = np.mean(loss)
    state = model.make_state()
    model.writer.add_scalar("Validation_loss", loss, model.counter['batches'])
    model.writer.add_scalar("Validation_acc", accuracy, model.counter['batches'])
    model.update_learning_rate(loss)
    model.early_stopping(loss, state)
    
def get_value(tensor):
    return tensor.detach().cpu().numpy()

def main():
    args = get_parser().parse_args()
    # Arguments by hand
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.target_name = "LST_status"

    # Initilize device

    # Initialize dataloader Creates 2 dataset : Careful, if I want to load all in memory ill have to change that, to have only one dataset.
    dataloader_train, dataloader_val = make_loaders(args=args)
    model = Classifier(args=args)
    model.dataset = dataloader_train.dataset.files

    while model.counter['epochs'] < args.epochs:
        print("Begin training")
        train(model=model, dataloader=dataloader_train)
        val(model=model, dataloader=dataloader_val)
        if model.early_stopping.early_stop:
            break
    model.writer.close()
        
if __name__ == "__main__":
    main()
