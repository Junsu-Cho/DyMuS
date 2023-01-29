import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import random
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True):
        super(GRU, self).__init__()

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        self.gru_cell = GRUCell(input_dim, hidden_dim)

    def forward(self, x):
        # x: b * l * d
        
        # b * d
        h0 = torch.zeros((x.size(0), self.hidden_dim), device=device)
       
        # b * d
        hn = h0
        for seq in range(x.size(1)):
            hn = self.gru_cell(x[:,seq,:], hn) 
        out = hn 

        return out


class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True):
        super(GRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.bias = bias
        self.Wi = nn.Linear(input_dim, 3 * hidden_dim, bias=bias)
        self.Wh = nn.Linear(hidden_dim, 3 * hidden_dim, bias=bias)
    
    def forward(self, x, hk_1):
        # x: b * d
        # hk_1: b * d

        batch_size = x.shape[0]
        x = x.reshape(-1, x.shape[-1])
                
        # b * (3*d)
        W_i = self.Wi(x) 
        W_h = self.Wh(hk_1)
                            
        # b * d
        W_ir_i, W_iz_i, W_in_i = W_i.chunk(3, dim=-1) 
        W_hr_h, W_hz_h, W_hn_h = W_h.chunk(3, dim=-1)

        # b * c * d2
        r = torch.sigmoid(W_ir_i + W_hr_h)
        z = torch.sigmoid(W_iz_i + W_hz_h)
        n = torch.tanh(W_in_i + (r * W_hn_h))

        hy = (z * n + (1.0 - z) * hk_1) 
        
        return hy



class dynamicGRU(nn.Module):
    def __init__(self, num_beh, input_dim, num_classes, capsule_length, bias=True):
        super(dynamicGRU, self).__init__()
        # Hidden dimensions
        self.num_beh = num_beh
        self.hidden_dim = num_classes * capsule_length
        self.num_classes = num_classes
        self.capsule_length = capsule_length

        self.gru_cell = dynamicGRUCell(num_beh, input_dim, num_classes, capsule_length)
    
    
    def forward(self, x, c, detach=False):
        # x: b * t * l * d
        # c: b * t * l * c * d2

        length = x.shape[2]
        H0 = torch.zeros((x.size(0), self.num_beh, self.num_classes, self.capsule_length), device=device)
       
        Ns = []

        # b * t * c * d2
        Hk = H0
        for k in range(length):
            # b * t * c * d2, b * t * c * d2
            Hk, N = self.gru_cell(x[:,:,k,:], Hk, c[:,:,k,:,:], detach=detach) 
            Ns.append(N)

        # b * t * c * d2
        out = Hk

        # b * t * l * c * d2
        Ns = torch.stack(Ns, dim=2) 
        
        # b * t * c * d2, b * t * l * c * d2
        return out, Ns


class dynamicGRUCell(nn.Module):
    def __init__(self, num_beh, input_dim, num_classes, capsule_length, bias=True):
        super(dynamicGRUCell, self).__init__()
        self.num_beh = num_beh
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.capsule_length = capsule_length
        self.bias = bias
        
        x_std = np.sqrt(1.0 / input_dim)
        h_std = np.sqrt(1.0 / capsule_length)


        # t * d * (c*3*d2)
        self.Wi = nn.Parameter(torch.rand((num_beh, input_dim, num_classes, 3 * capsule_length)) * 2*x_std - x_std)
        # t * d2 * (3*d2)
        self.Wh = nn.Parameter(torch.rand((num_beh, num_classes, capsule_length, 3 * capsule_length)) * 2*h_std - h_std)
        self.Wc = nn.Parameter(torch.rand((num_beh, num_classes, capsule_length, 3 * capsule_length)) * 2*h_std - h_std)

        self.Bx = nn.Parameter(torch.rand((num_beh, num_classes, 3 * capsule_length)) * 2 * x_std - x_std)
        self.Bh = nn.Parameter(torch.rand((num_beh, num_classes, 3 * capsule_length)) * 2 * h_std - h_std)
        self.Bc = nn.Parameter(torch.rand((num_beh, num_classes, 3 * capsule_length)) * 2 * h_std - h_std)


    def forward(self, x, Hk_1, C, detach=False):
        # x: b * t * d
        # Hk_1: b * t * c * d2
        # C: b * t * c * d2

        if detach:         
            # b * t * c * (3*d2)
            W_i = torch.einsum('bti,tick->btck', x.detach(), self.Wi.detach()) + self.Bx.detach().unsqueeze(0)
            W_h = torch.einsum('btcd,tcdl->btcl', Hk_1.detach(), self.Wh.detach()) + self.Bh.detach().unsqueeze(0)
            W_c = torch.einsum('btcd,tcdl->btcl', C.detach(), self.Wc.detach()) + self.Bc.detach().unsqueeze(0)
            
        else:
            # b * t * c * (3*d2)
            W_i = torch.einsum('bti,tick->btck', x, self.Wi) + self.Bx.unsqueeze(0)
            W_h = torch.einsum('btcd,tcdl->btcl', Hk_1, self.Wh) + self.Bh.unsqueeze(0)
            W_c = torch.einsum('btcd,tcdl->btcl', C, self.Wc) + self.Bc.unsqueeze(0)
                            
        # b * t * c * d2
        W_ir_i, W_iz_i, W_in_i = W_i.chunk(3, dim=-1) 
        W_hr_H, W_hz_H, W_hn_H = W_h.chunk(3, dim=-1)
        W_cr_C, W_cz_C, W_cn_C = W_c.chunk(3, dim=-1)

        W_ir_i = W_ir_i * W_cr_C
        W_iz_i = W_iz_i * W_cz_C
        W_in_i = W_in_i * W_cn_C

        # b * t * c * d2
        R = torch.sigmoid(W_ir_i + W_hr_H)
        Z = torch.sigmoid(W_iz_i + W_hz_H)
        N = torch.tanh(W_in_i + (R * W_hn_H) )

        H = (Z * N + (1.0 - Z) * Hk_1) 
        
        return H, N