from argparse import ArgumentParser
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch
import sklearn.metrics as metrics
import numpy as np
# Local imports
from arguments import get_parser
from models import instanciate_model
from dataloader import make_loaders

def train(model, dataloader):
    model.network.train()
    for input_batch, target_batch in dataloader:
        model.counter['batches'] += 1
        loss = model.optimize_parameters(input_batch, target_batch)
        mean_loss = get_value(loss)
        model.writer.add_scalars("Training", {'loss' :mean_loss}, model.counter['batches'])
    model.counter['epochs'] += 1

def val(model, dataloader):
    model.network.eval()
    accuracy = []
    loss = []
    for input_batch, target_batch in dataloader:
        output, preds = model.predict(input_batch)
        loss.append(model.criterion(output, target_batch).detach().numpy())
        accuracy.append(metrics.accuracy_score(y_true=np.array(target_batch.numpy()), y_pred=preds))
    accuracy = np.mean(accuracy)
    loss = np.mean(loss)
    state = model.make_state()
    model.writer.add_scalars("Validation", {'loss': loss, 'accuracy': accuracy}, model.counter['batches'])
    model.update_learning_rate(loss)
    model.early_stopping(loss, state)
    
def get_value(tensor):
    return tensor.detach().cpu().numpy()

def main():
    args = get_parser().parse_args()
    # Initialize seed

    # Initilize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize dataloader Creates 2 dataset : Careful, if I want to load all in memory ill have to change that, to have only one dataset.
    dataloader_train, dataloader_val = make_loaders(args=args)
    model = instanciate_model(args=args, device=device)

    while model.counter['epochs'] < args.epochs:
        print("Begin training")
        train(model=model, dataloader=dataloader_train)
        val(model=model, dataloader=dataloader_val)
        if model.early_stopping.early_stop:
            break
    model.writer.close()
        
if __name__ == "__main__":
    main()