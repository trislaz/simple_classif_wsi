from model_base import Model
from torchvision import models
import numpy as np
import torch
from tensorboardX import SummaryWriter

def instanciate_model(args, device):
    if args.model == 'resnet18':
        model = ResNet18(args, device)
    return model

class ResNet18(Model):
    def __init__(self, args, device):
        super(ResNet18, self).__init__(args, device)
        self.network = models.resnet18()
        self.network.fc = torch.nn.Linear(in_features=512, out_features=2)
        self.network.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.writer = SummaryWriter('./')
        optimizer = torch.optim.Adam(self.network.parameters())
        self.optimizers = [optimizer]
        self.get_schedulers()

    def optimize_parameters(self, input_batch, target_batch):
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        output_batch = self.forward(input_batch)
        self.set_zero_grad()
        loss = self.criterion(output_batch, target_batch)
        loss.backward()
        return loss

    def forward(self, x):
        return self.network(x)
    
    def predict(self, x):
        output = self.forward(x)
        proba = torch.nn.functional.softmax(output, dim=1)
        _, preds = torch.max(proba, dim=1)
        proba_positif = proba[:,1]
        assert [round(x) for x in proba_positif.detach().numpy()] == list(preds)
        return output, np.array(preds.detach().numpy())

    def make_state(self):
        dictio = {'state_dict': self.network.state_dict(),
                  'state_dict_optimizer': self.optimizers[0].state_dict, 
                  'state_scheduler': self.schedulers[0].state_dict(), 
                  'inner_counter': self.counter}
        return dictio
