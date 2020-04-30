from argparse import ArgumentParser
from torch.utils.data import DataLoader, SubsetRandomSampler
import os
import pandas as pd
import torch
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold
import numpy as np
# Local imports
from arguments import get_parser
from models import Classifier 
from dataloader import make_loaders, dataset, get_transform

def train(model, dataloader):
    model.network.train()
    for input_batch, target_batch in dataloader:
        model.counter['batches'] += 1
        loss = model.optimize_parameters(input_batch, target_batch)
        mean_loss = get_value(loss)
        model.writer.add_scalar("Training_loss_{}".format(model.name), mean_loss, model.counter['batches'])
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
    model.writer.add_scalar("Validation_loss_{}".format(model.name), loss, model.counter['batches'])
    model.writer.add_scalar("Validation_acc_{}".format(model.name), accuracy, model.counter['batches'])
    model.update_learning_rate(loss)
    model.early_stopping(loss, state, model.name)
    
def get_value(tensor):
    return tensor.detach().cpu().numpy()

def main():
    args = get_parser().parse_args()
    # Arguments by hand
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.target_name = "LST_status"

    table = pd.read_csv(args.table_data)
    list_wsi = os.listdir(args.wsi)
    list_lst = [table[table['ID'] == x][args.target_name].item() for x in list_wsi]
    list_dataset = []

    ## Initialisation model
    model = Classifier(args=args)

    ## Création des datasets
    for path in list_wsi:
        args.wsi = os.path.join(args.wsi, path)
        list_dataset.append(dataset(args))
        args.wsi = os.path.dirname(args.wsi)
    list_dataset = np.array(list_dataset)

    ## Kfold_validation
    splitter = StratifiedKFold(n_splits=3)
    for r_eval, (id_train, id_val) in enumerate(splitter.split(list_lst, list_lst)):

        model.name = 'repeat_val_{}'.format(r_eval)
        dataset_train = list_dataset[id_train]
        dataset_val = list_dataset[id_val]
        for db in dataset_train:
            db.transform = get_transform(train=True)
        for db in dataset_val:
            db.transform = get_transform(train=False)
        dataset_train = torch.utils.data.ConcatDataset(dataset_train)
        dataset_val = torch.utils.data.ConcatDataset(dataset_val)
        dataloader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, num_workers=24)
        dataloader_val = DataLoader(dataset=dataset_val, batch_size=args.batch_size, num_workers=24)

        # Initialize dataloader Creates 2 dataset : Careful, if I want to load all in memory ill have to change that, to have only one dataset.
        dataloader_train, dataloader_val = make_loaders(args=args)

        while model.counter['epochs'] < args.epochs:
            print("Begin training")
            train(model=model, dataloader=dataloader_train)
            val(model=model, dataloader=dataloader_val)
            if model.early_stopping.early_stop:
                break
        model.writer.close()

#%%
if __name__ == "__main__":
    main()
    #from argparse import Namespace
    #from dataloader import dataset, get_transform
    #args = Namespace()
    #args.wsi = "/Users/trislaz/Documents/cbio/data/tcga/tiled_tumor_res1/image_tcga_1"
    #args.n_sample = 5
    #args.target_name = "LST_status"
    #args.table_data = "../gradCAM/data/labels_tcga_tnbc_strat.csv"
    #dataset = dataset(args, transform=get_transform(train=False))

# %%
