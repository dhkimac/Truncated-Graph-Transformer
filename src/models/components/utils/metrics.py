from torchmetrics import Metric
import torch 
from .loss_func import RateLoss

class SumRate(Metric):
    def __init__(self):
        super().__init__()
        self.rate_loss = RateLoss(1.0)
        self.add_state("sum_rate", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_graphs", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self,graph, allocs,noise):
        self.rate_loss.noise = noise
        rates = self.rate_loss(graph,allocs,no_mean=True)
        self.sum_rate += torch.sum(rates)
        self.num_samples += rates.shape[0]
        self.num_graphs += graph.batch_size
    
    def compute(self):
        return self.sum_rate / self.num_graphs
