#%%
import torch 
import torchvision
from abc import ABC, abstractmethod
import shutil


class Model(ABC):

    def __init__(self, args, device):

        self.args = args
        self.optimizers = []
        self.losses = {'global':[]}
        self.metric = 0
        self.criterion = lambda : 1
        self.counter = {'epochs': 0, 'batches': 0}
        self.network = torch.nn.Module()
        self.early_stopping = EarlyStopping(args=args)
        self.device = device

    @abstractmethod
    def optimize_parameters(self, input_batch, target_batch):
        pass

    @abstractmethod
    def make_state(self):
        pass

    def get_schedulers(self):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
        self.schedulers = [scheduler(optimizer=o, patience=100) for o in self.optimizers]

    def update_learning_rate(self, metric):
        for sch in self.schedulers:
            sch.step(metric)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def set_zero_grad(self):
        optimizers = self.optimizers
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            if optimizer is not None:
                optimizer.zero_grad()


class EarlyStopping:
    """Early stopping AND saver !
    """
    def __init__(self,args):
        self.patience = args.patience
        self.counter = 0
        self.loss_min = None
        self.early_stop = False
        self.is_best = False
        self.filename = 'model.pt.tar'

    def __call__(self, loss, state):

        #####################
        ## Learning begins ##
        #####################
        if self.loss_min is None:
            self.loss_min = loss
            self.is_best = True

        #####################
        ## No progression ###
        #####################
        elif loss > self.loss_min:
            self.counter += 1
            self.is_best = False
            if self.counter < self.patience:
                print('-- There has been {}/{} epochs without improvement on the validation set. --\n'.format(
                      self.counter, self.patience))
            else:
                self.early_stop = True

        ######################
        ## Learning WIP ######
        ######################
        else:
            self.is_best = True
            self.loss_min = loss
            self.counter = 0
        
        self.save_checkpoint(state)

    def save_checkpoint(self, state):
        torch.save(state, self.filename)
        if self.is_best:
            shutil.copyfile(self.filename, self.filename.replace('.pt.tar','_best.pt.tar'))