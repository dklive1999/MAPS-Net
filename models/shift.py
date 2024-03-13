import torch

def shift(x, gamma=1/12):
    # feat is a tensor with a shape of
    # [Batch, Channel, Height, Width]
    B, C, H, W = x.shape
    g = int(gamma * C)
    out = torch.zeros_like(x)
    # spatially shift
    out[:, 0*g:1*g, :, :-1] = x[:, 0*g:1*g, :, 1:] # shift left
    out[:, 1*g:2*g, :, 1:] = x[:, 1*g:2*g, :, :-1] # shift right
    out[:, 2*g:3*g, :-1, :] = x[:, 2*g:3*g, 1:, :] # shift up
    out[:, 3*g:4*g, 1:, :] = x[:, 3*g:4*g, :-1, :] # shift down
    # remaining channels
    out[:, 4*g:, :, :] = x[:, 4*g:, :, :] # no shift
    return out
