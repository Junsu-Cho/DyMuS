# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import random
import numpy as np
from model_util import *

import time

torch.set_printoptions(edgeitems=100)


class DyMuS(nn.Module):
    def __init__(self, num_beh, num_items, embedding_dim, max_purchase_length, max_cart_length, max_fav_length, max_click_length, r, capsule_length, dropout):
        super(DyMuS, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_items = num_items
        self.embedding_dim = embedding_dim

        self.max_purchase_length = max_purchase_length
        self.max_cart_length = max_cart_length
        self.max_fav_length = max_fav_length
        self.max_click_length = max_click_length
        self.dropout = nn.Dropout(dropout)

        self.embedding_dim = embedding_dim
        self.hidden_dim = self.embedding_dim
        self.num_classes = self.embedding_dim 
        
        self.item_embedding = nn.Embedding(self.num_items + 1, self.embedding_dim, padding_idx = 0)
        

        ## Parameters
        self.r = r
        self.num_beh = num_beh

        
        self.capsule_length = capsule_length
        
        self.W = nn.Parameter(0.01 * torch.rand(self.hidden_dim, self.num_beh, self.num_classes, self.capsule_length))

        ## click encoder: GRU
        self.purchase_gru = GRU(self.embedding_dim, self.hidden_dim)

        ## click encoder: GRU
        self.click_gru = GRU(self.embedding_dim, self.hidden_dim)

        ## Cart encoder: GRU
        self.cart_gru = GRU(self.embedding_dim, self.hidden_dim)

        ## Fav encoder: GRU
        self.fav_gru = GRU(self.embedding_dim, self.hidden_dim)

        self.w = nn.Parameter(torch.ones(self.num_classes).to(self.device) )
        self.bias = nn.Parameter(torch.zeros(self.num_classes).to(self.device))
        self.alpha = nn.Parameter(torch.ones(1) * self.num_classes)
        self.W_coef = nn.Parameter(0.1 * torch.randn(self.num_classes, self.embedding_dim + self.capsule_length, self.capsule_length))


    def forward(self, purchase_seq, cart_seq, fav_seq, click_seq):
        # purchase, cart, fav, click_seq: b * l

        assert purchase_seq.size()[0] == click_seq.size()[0]
        assert purchase_seq.size()[0] == cart_seq.size()[0]
        assert purchase_seq.size()[0] == fav_seq.size()[0]
        
        batch_size = purchase_seq.shape[0]

        max_purchase_length = self.max_purchase_length
        max_cart_length = self.max_cart_length
        max_fav_length = self.max_fav_length
        max_click_length = self.max_click_length

        embedding_dim = self.embedding_dim
        num_beh = self.num_beh
        num_classes = self.num_classes
        capsule_length = self.capsule_length

        # b * l * 1
        valid_purchase_seq = (purchase_seq != 0.0).to(torch.float).unsqueeze(-1) 
        valid_cart_seq = (cart_seq != 0.0).to(torch.float).unsqueeze(-1) 
        valid_fav_seq = (fav_seq != 0.0).to(torch.float).unsqueeze(-1) 
        valid_click_seq = (click_seq != 0.0).to(torch.float).unsqueeze(-1) 

        # b * 1
        valid_purchase = (torch.sum(valid_purchase_seq, dim=1) != 0.0).to(torch.float)
        valid_cart = (torch.sum(valid_cart_seq, dim=1) != 0.0).to(torch.float)
        valid_fav = (torch.sum(valid_fav_seq, dim=1) != 0.0).to(torch.float)
        valid_click = (torch.sum(valid_click_seq, dim=1) != 0.0).to(torch.float)
        
        ## Build embedding
        # Input embeddings
        # b * l * d
        purchaseEmbeddings = self.item_embedding(purchase_seq) #** 2
        cartEmbeddings = self.item_embedding(cart_seq) #** 2
        favEmbeddings = self.item_embedding(fav_seq) #** 2
        clickEmbeddings = self.item_embedding(click_seq) #** 2

        # b * l * d
        purchaseEmbeddings = purchaseEmbeddings * valid_purchase_seq 
        cartEmbeddings = cartEmbeddings * valid_cart_seq 
        favEmbeddings = favEmbeddings * valid_fav_seq 
        clickEmbeddings = clickEmbeddings * valid_click_seq 
        
        # Embedding for prediction
        # i * d
        itemEmbeddings = self.item_embedding.weight
        

        ## GRU
        # b * d 
        h_purchase = self.purchase_gru(purchaseEmbeddings)
        h_cart = self.cart_gru(cartEmbeddings)
        h_fav = self.fav_gru(favEmbeddings)
        h_click = self.click_gru(clickEmbeddings)
        
        ## Dynamic routing
        # b * t * h
        e = torch.stack([h_purchase, h_cart, h_fav, h_click], dim=1)    

        
        # b * h * t
        e = e.permute(0, 2, 1) 
        e = self.dropout(e)

        # b * d * c * d2
        u = torch.einsum('bht,htcd->bhcd', e.reshape(batch_size, self.hidden_dim, self.num_beh), self.dropout(self.W))
        
        
        # b * h * c * 1
        b = torch.zeros((batch_size, self.hidden_dim, self.num_classes, 1), device=self.device)
                
        # b * h * c * d2
        u_detach = u.detach()
        for i in range(self.r-1):
            # b * h * c * 1
            c = torch.softmax(b, dim=2)
            
            # b * c * d2
            v = self.alpha.detach() * torch.sum(c * u_detach, dim=1) # b * t * c

            # b * c
            v_ = torch.norm(v, dim=-1) * self.w.detach() + self.bias.detach() # bc

            # b * h * c * d 
            p = torch.matmul(torch.softmax(torch.matmul(v_, itemEmbeddings.detach().t()), dim=-1), itemEmbeddings.detach()) # b * d
            p = p.unsqueeze(1).repeat(1, self.num_classes, 1) # b * t * c 

            # b * t * c * 1
            r = torch.sum(u_detach * torch.einsum('bcd,cdD->bcD', torch.cat([p, v], dim=-1), self.W_coef).unsqueeze(1), dim=-1, keepdim=True)

            b = b + r

        # b * h * c * 1
        c = torch.softmax(b, dim=2)

        # b * c * d2
        v = self.alpha * torch.sum(c * u, dim=1)

        # b * c
        v_ = torch.norm(v, dim=-1) * self.w + self.bias

        # b * o * i
        result = torch.matmul(v_, itemEmbeddings.t())
        
        return result

        

class DyMuS_plus(nn.Module):
    def __init__(self, num_beh, num_items, embedding_dim, max_purchase_length, max_cart_length, max_fav_length, max_click_length, r, capsule_length, dropout):
        super(DyMuS_plus, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_items = num_items
        self.embedding_dim = embedding_dim

        self.max_purchase_length = max_purchase_length
        self.max_cart_length = max_cart_length
        self.max_fav_length = max_fav_length
        self.max_click_length = max_click_length
        self.dropout = nn.Dropout(dropout)

        self.item_embedding = nn.Embedding(self.num_items + 1, self.embedding_dim, padding_idx = 0)

        ## Parameters
        self.r = r
        self.num_beh = num_beh
        self.num_classes = self.embedding_dim
        self.capsule_length = capsule_length


        ## Weights
        self.W = nn.Parameter(0.01 * torch.rand(self.num_classes, self.capsule_length, self.num_beh, self.num_classes))
        self.W_coef = nn.Parameter(0.1 * torch.randn(self.num_classes, self.embedding_dim + self.capsule_length, self.capsule_length))

        self.alpha = nn.Parameter(torch.ones(1) * self.num_classes)
        self.w = nn.Parameter(torch.ones(self.num_classes).to(self.device))
        self.bias = nn.Parameter(torch.zeros(self.num_classes).to(self.device))


        ## Dynamic GRU
        self.beh_gru = dynamicGRU(self.num_beh, self.embedding_dim, self.num_classes, self.capsule_length, bias=True)


    def forward(self, purchase_seq, cart_seq, fav_seq, click_seq):
        # purchase, cart, fav, click_seq: b * l

        assert purchase_seq.size()[0] == click_seq.size()[0]
        assert purchase_seq.size()[0] == cart_seq.size()[0]
        assert purchase_seq.size()[0] == fav_seq.size()[0]
        
        batch_size = purchase_seq.shape[0]

        max_purchase_length = self.max_purchase_length
        max_cart_length = self.max_cart_length
        max_fav_length = self.max_fav_length
        max_click_length = self.max_click_length

        embedding_dim = self.embedding_dim
        num_beh = self.num_beh
        num_classes = self.num_classes
        capsule_length = self.capsule_length

        # b * l * 1
        valid_purchase_seq = (purchase_seq != 0.0).to(torch.float).unsqueeze(-1) 
        valid_cart_seq = (cart_seq != 0.0).to(torch.float).unsqueeze(-1) 
        valid_fav_seq = (fav_seq != 0.0).to(torch.float).unsqueeze(-1) 
        valid_click_seq = (click_seq != 0.0).to(torch.float).unsqueeze(-1) 

        ## Build embedding
        # Input embeddings
        # b * l * d
        purchaseEmbeddings = self.item_embedding(purchase_seq) 
        cartEmbeddings = self.item_embedding(cart_seq) 
        favEmbeddings = self.item_embedding(fav_seq) 
        clickEmbeddings = self.item_embedding(click_seq) 

        # b * l * d
        purchaseEmbeddings = purchaseEmbeddings * valid_purchase_seq 
        cartEmbeddings = cartEmbeddings * valid_cart_seq 
        favEmbeddings = favEmbeddings * valid_fav_seq 
        clickEmbeddings = clickEmbeddings * valid_click_seq 
        
        # Embedding for prediction
        # i * d
        itemEmbeddings = self.item_embedding.weight
        

        ## Dynamic GRU
        # b * h * c * 1
        b = torch.zeros((batch_size, num_classes, num_classes, 1), device=self.device)

        # b * t * l * c * d2
        C = torch.zeros((batch_size, num_beh, max_purchase_length, num_classes, capsule_length), device=self.device)            

        assert max_purchase_length == max_cart_length
        assert max_purchase_length == max_fav_length
        assert max_purchase_length == max_click_length

        # b * t * l * d
        behEmbeddings = torch.stack([purchaseEmbeddings, cartEmbeddings, favEmbeddings, clickEmbeddings], dim=1)
        behEmbeddings_detach = behEmbeddings.detach()

        for i in range(self.r - 1):
            # b * t * c * d2, b * t * l * c * d2
            beh_e, beh_N = self.beh_gru(behEmbeddings_detach, C, detach=True)
            beh_e = self.dropout(beh_e)
            beh_N = self.dropout(beh_N)

            # b * c * t * d2
            beh_e = beh_e.permute(0, 2, 1, 3)
            
            ## Dynamic routing
            # b * c * c * d2
            u_detach = torch.einsum('bctd,cdtC->bcCd', beh_e.detach(), self.dropout(self.W).detach())

            # b * h * c * 1
            c = torch.softmax(b, dim=2)
            
            # b * c * d2
            v = self.alpha.detach() * torch.sum(c * u_detach, dim=1) 

            # b * c
            v_ = torch.norm(v, dim=-1) * self.w.detach() + self.bias.detach()

            # b * d
            p = torch.matmul(torch.softmax(torch.matmul(v_, itemEmbeddings.detach().t()), dim=-1), itemEmbeddings.detach()) # b * d            
            p = p.unsqueeze(1).repeat(1, self.num_classes, 1) # b * t * c 

            # b * c * d2
            r = torch.einsum('bcd,cdD->bcD', torch.cat([p, v], dim=-1), self.W_coef)

            # b * t * c * 1
            b = b + torch.sum(u_detach * r.unsqueeze(1), dim=-1, keepdim=True)
            
            # b * t * l * c * d2
            C = C + beh_N.detach() * r.unsqueeze(1).unsqueeze(1)

        # b * t * c * d2
        beh_e, _ = self.beh_gru(behEmbeddings, C)
        beh_e = self.dropout(beh_e)

        # b * c * t * d2
        beh_e = beh_e.permute(0, 2, 1, 3)

        # b * t * c * d2
        u_hat = torch.einsum('bctd,cdtC->bcCd', beh_e, self.dropout(self.W))

        # b * t * c * 1
        c = torch.softmax(b, dim=2)
        
        # b * c * d2
        v = self.alpha * torch.sum(c * u_hat, dim=1) 

        # b * c
        v_ = torch.norm(v, dim=-1) * self.w + self.bias

        # b * i
        result = torch.matmul(v_, itemEmbeddings.t()) 

        return result


