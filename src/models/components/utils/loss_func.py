import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
import torch
import dgl


def rate_loss(noise, allocs, path_loss,no_mean = False):
    """
    Average Sumrate of all nodes.
    :param allocs: Power Allocation of all nodes.
    :param directlink_channel_losses: Direct Link Channel Losses of transmitter reciever pairs
    :param crosslink_channel_losses: Cross Link Channel Losses of other pairs
    """
    path_loss = torch.square(path_loss.float())
    directlink_channel_losses = torch.diagonal(path_loss, dim1=1, dim2=2) 
    crosslink_channel_losses = path_loss* ((torch.eye(path_loss.shape[-1]) < 1).float()).to("cuda")
    SINRs_numerators = allocs * directlink_channel_losses
    SINRs_denominators = torch.squeeze(torch.matmul(crosslink_channel_losses, torch.unsqueeze(allocs, axis=-1))) + noise
    SINRs = SINRs_numerators / SINRs_denominators
    rates = torch.log2(1 + SINRs)
    if no_mean:
        return torch.sum(rates, axis = -1)
    return -torch.mean(torch.sum(rates, axis = -1))

class RateLoss(torch.nn.Module):
    def __init__(self, noise):
        super(RateLoss, self).__init__()
        self.noise = noise
    def message_func(self, edges):
        interference = edges.src['allocs'] * torch.square(edges.data['csi']).unsqueeze(-1)
        return {'interference': interference}
    
    def reduce_func(self, nodes):
        interference = nodes.mailbox['interference']
        interference = torch.sum(interference, dim=1)
        interference = interference + self.noise
        rate = torch.log2(1 + torch.square(nodes.data['csi'])*nodes.data['allocs'] / interference)

        return {'rate': rate}

    def forward(self, g, allocs, no_mean=False):
        g_n = dgl.remove_self_loop(g)
        g_n.ndata['allocs'] = allocs
        g_n.update_all(self.message_func, self.reduce_func)
        if no_mean:
            return g_n.ndata['rate']
        return -torch.mean(g_n.ndata['rate'])

class RateLossWeight(torch.nn.Module):
    def __init__(self, noise):
        super(RateLossWeight, self).__init__()
        self.noise = noise
    def message_func(self, edges):
        interference = edges.src['allocs'] * torch.square(edges.data['csi']).unsqueeze(-1)
        return {'interference': interference}
    
    def reduce_func(self, nodes):
        interference = nodes.mailbox['interference']
        interference = torch.sum(interference, dim=1)
        interference = interference + self.noise
        rate = torch.log2(1 + torch.square(nodes.data['csi'])*nodes.data['allocs'] / interference)

        return {'rate': rate}

    def forward(self, g, allocs,weights, no_mean=False):
        g_n = dgl.remove_self_loop(g)
        g_n.ndata['allocs'] = allocs
        g_n.update_all(self.message_func, self.reduce_func)
        g_n.ndata['rate'] = g_n.ndata['rate']*weights.unsqueeze(-1)
        if no_mean:
            return g_n.ndata['rate']
        return -torch.mean(g_n.ndata['rate'])