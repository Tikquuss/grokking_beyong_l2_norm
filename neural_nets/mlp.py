import torch
import torch.nn as nn
import numpy as np

import math

from typing import List, Dict, Tuple, Set, Optional, Union, Any
import itertools

########################################################################################
############################################ FNN #######################################
########################################################################################

class Feed_Forward_Neural_Network(nn.Module):
    """
    Feed Forward Neural Network
    """
    def __init__(self):
        super(Feed_Forward_Neural_Network, self).__init__()

    def get_representation(self, x) :
        """
        x : (batch_size, input_dim*)
        """
        raise NotImplementedError("get_representation() must be implemented in subclasses.")

    def get_logits(self, representation) :
        """
        representation : (batch_size, rep_dim*)
        """
        raise NotImplementedError("get_logits() must be implemented in subclasses.")

    def forward(self, x, activation=False):
        """
        x : (batch size, input_dim*)
        """
        h = self.get_representation(x) # (batch_size, rep_dim*)
        logits = self.get_logits(h) # (batch_size, output_dim*)
        if activation : return logits, h
        return logits

def initialize_weights(cls, init_params, widths, readout, fc=None, type_init='normal', init_scale=None, last_layer_zero_init=False, seed=None):
    """
    Initialize the weights of each trainable layer of your network using Normal or Kaming/Xavier uniform initialization.
    """
    assert type_init in ['kaiming', 'xavier', 'normal']

    if seed is not None:
        # Save the current random state
        torch_state = torch.get_rng_state()
        torch.manual_seed(seed)

    if init_params:
        with torch.no_grad():
            if len(widths)>=3:
                for layer in readout:
                    if isinstance(layer, nn.Linear):
                        if type_init=="normal":
                            torch.nn.init.normal_(layer.weight, mean=0.0, std=1.0 / math.sqrt(layer.weight.shape[0]))
                            # torch.nn.init.normal_(layer.weight, mean=0.0, std=0.25**0.5 / np.power(2*layer.weight.shape[0], 1/3))
                        elif type_init=="kaiming":
                            nn.init.kaiming_uniform_(layer.weight)
                        elif type_init=="xavier":
                            nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)

            if isinstance(fc, nn.Linear):
                if type_init=="normal":
                    torch.nn.init.normal_(fc.weight, mean=0.0, std=1.0 / math.sqrt(fc.weight.shape[0]))
                    # torch.nn.init.normal_(fc.weight, mean=0.0, std=0.25**0.5 / np.power(2*fc.weight.shape[0], 1/3))
                elif type_init=="kaiming":
                    nn.init.kaiming_uniform_(fc.weight)
                elif type_init=="xavier":
                    nn.init.xavier_uniform_(fc.weight)
                if fc.bias is not None:
                    nn.init.constant_(fc.bias, 0)

    if (init_scale!=1.0) and (init_scale is not None):
        with torch.no_grad():
            for p in cls.parameters():
                p.data = init_scale * p.data

    if last_layer_zero_init and isinstance(fc, nn.Linear):
        fc.weight.data.zero_()
        if fc.bias is not None:
            fc.bias.data.zero_()

    # Restore the previous random state
    if seed is not None:
        torch.set_rng_state(torch_state)

########################################################################################
############################################ MLP #######################################
########################################################################################

def make_mlp(widths, activation_class=None, bias=True, head=[], tail=[]):
    L = len(widths)
    if L <= 1 : return torch.nn.Identity()
    return torch.nn.Sequential(*(head + sum(
        [[torch.nn.Linear(i, o, bias=bias)] + ([activation_class()] if (n < L-2 and activation_class is not None) else [])
         for n, (i, o) in enumerate(zip(widths, widths[1:]))], []) + tail))

class MLP(Feed_Forward_Neural_Network):
    """
    MultiLayer Perceptron
    """
    def __init__(
        self, widths, activation_class=nn.ReLU, bias=True, bias_classifier=True, dropout=0.0, 
        init_scale=None, init_params=False, type_init='normal', last_layer_zero_init=False, seed=None):

        super(MLP, self).__init__()

        #assert len(widths) >= 3
        assert len(widths) >= 2
        self.widths = widths

        # Input transformation
        self.input_transform = nn.Flatten()

        self.readout = make_mlp(
            widths=widths[:-1], activation_class=activation_class, bias=bias, head=[], 
            tail=([nn.Dropout(p=dropout)] if dropout>0 else []) + ([activation_class()] if activation_class is not None else []))
    
        self.fc = nn.Linear(widths[-2], widths[-1], bias=bias_classifier)
        #self.fc = LinearClassifier(widths[-2], widths[-1], bias=bias_classifier)

        initialize_weights(self, init_params, widths, self.readout, self.fc, type_init, init_scale, last_layer_zero_init, seed)

    def get_representation(self, x) :
        """
        x : (batch_size, input_dim*)
        """
        return self.readout( self.input_transform(x) ) # (batch_size, rep_dim*)

    def get_logits(self, representation) :
        """
        representation : (batch_size, rep_dim*)
        """
        return self.fc(representation) # (batch_size, output_dim*)


########################################################################################
###################################### Encoder Decoder #################################
########################################################################################


class Encoder_Decoder(Feed_Forward_Neural_Network):
    """
    Encoder Decoder architecture
    """
    def __init__(
        self, aggregation_mode:str, widths_encoder:List, widths_decoder:List, activation_class_encoder=None, activation_class_decoder=nn.ReLU, 
        bias_encoder:bool=True, bias_decoder:bool=True, bias_classifier:bool=True, 
        dropout:float=0.0, init_scale:float=None, init_params:bool=False, type_init:str='normal', seed:int=None, last_layer_zero_init:bool=False):
        super(Encoder_Decoder, self).__init__()

        assert widths_encoder[-1] == widths_decoder[0]
        assert aggregation_mode in ['sum', 'concat', 'matrix_product', 'hadamard_product']

        
        rep_dim = widths_encoder[-1]
        if aggregation_mode == 'concat':    
            widths_decoder[0] = 2 * widths_decoder[0]   
        elif aggregation_mode == 'matrix_product':  
            s = 2  
            self.rep_dim_sqrt = int(rep_dim**(1/s))
            assert self.rep_dim_sqrt**s == rep_dim, f"rep_dim ({rep_dim}) must be a perfect square for {aggregation_mode} mode."

        self.aggregation_mode = aggregation_mode
        self.activation_class_encoder = activation_class_encoder
        self.widths_encoder = widths_encoder
        self.widths_decoder = widths_decoder

        # Encoders
        self.encoder = nn.ModuleList([
            make_mlp(
                widths=widths_encoder, activation_class=activation_class_encoder, bias=bias_encoder, head=[nn.Flatten()], 
                tail=([nn.Dropout(p=dropout)] if dropout>0 else []) + ([activation_class_encoder()] if activation_class_encoder is not None else [])) 
            for _ in range(2)])

        # Decoder = readout + fc
        self.readout = make_mlp(
            widths=widths_decoder[:-1], activation_class=activation_class_decoder, bias=bias_decoder, head=[], 
            tail=([nn.Dropout(p=dropout)] if dropout>0 else []) + ([activation_class_decoder()] if activation_class_decoder is not None else []))

        self.fc = nn.Linear(widths_decoder[-2], widths_decoder[-1], bias=bias_classifier)

 
        for encoder in self.encoder :
            initialize_weights(self, init_params, widths_encoder, encoder, None, type_init, None, last_layer_zero_init, seed)
        initialize_weights(self, init_params, widths_decoder, self.readout, self.fc, type_init, init_scale, last_layer_zero_init, seed)

    def aggregate(self, h):
        """
        h : 2 * (batch_size, rep_dim*)
        """
        if self.aggregation_mode == 'sum':
            h_aggreg = torch.stack(h, dim=0).sum(dim=0) # (batch_size, rep_dim*)
        elif self.aggregation_mode == 'concat':
            # Concatenate the tensors along the last dimension
            h_aggreg = torch.cat(h, dim=-1)  # (batch_size, 2 * rep_dim)
        elif self.aggregation_mode == 'matrix_product':
            # Product of all tensors 
            # Reshape each tensor to (batch_size, rep_dim_sqrt, rep_dim_sqrt)
            reshaped_h = [tensor.reshape(-1, self.rep_dim_sqrt, self.rep_dim_sqrt) for tensor in h]
            h_aggreg = torch.bmm(reshaped_h[0], reshaped_h[1]) # (batch_size, rep_dim_sqrt, rep_dim_sqrt)
            # Flatten the result back to (batch_size, rep_dim)
            h_aggreg = h_aggreg.flatten(1, -1) # (batch_size, rep_dim*)

        elif self.aggregation_mode == 'hadamard_product':
            # Hadamard (element-wise) product of all tensors 
            h_aggreg = h[0] * h[1]
        else:
            raise ValueError(f"Invalid aggregation_mode '{self.aggregation_mode}'.")

        return h_aggreg

    def tokens_to_embeddings(self, x):
        """
        x : (batch_size, 2, input_dim*)
        """
        assert x.dim() >= 3
        assert x.shape[1] == 2
        h = [self.encoder[i](x[:,i]) for i in range(2)] # 2 * (batch_size, rep_dim*)
        return self.aggregate(h) # (batch_size, rep_dim*)

    def get_representation(self, x) :
        """
        x : (batch_size, 2, input_dim*)
        """
        #return self.tokens_to_embeddings(x) # (batch_size, rep_dim*)
        return self.readout(self.tokens_to_embeddings(x)) # (batch_size, rep_dim*)

    def get_logits(self, representation) :
        """
        representation : (batch_size, rep_dim*)
        """
        #return self.fc(self.readout(representation)) # (batch_size, output_dim*)
        return self.fc(representation) # (batch_size, output_dim*)

########################################################################################
########################################################################################

if __name__ == "__main__":

    input_dim = 3
    output_dim = 3
    batch_size = 10

    ## MLP
    # widths = [input_dim, 2, 2, output_dim]
    # model = MLP(
    #     widths, activation_class=nn.ReLU, bias=True, bias_classifier=True, dropout=0.0, 
    #     init_scale=1.0, init_params=True, type_init='normal', last_layer_zero_init=True, seed=None)
    # print(model)
    # x = torch.randn(size=(batch_size, input_dim)) # (batch_size, input_dim)
    # print(x.shape)
    # logits, h = model(x, activation=True)
    # print(logits.shape, h.shape)

    ## Multilinear Encoder Decoder
    aggregation_mode = 'hadamard_product' # 'sum', 'concat', 'matrix_product', 'hadamard_product'
    widths_encoder = [input_dim, 4]
    widths_decoder = [4, 7, output_dim]
    model = Encoder_Decoder(
        aggregation_mode, widths_encoder, widths_decoder, 
        activation_class_encoder=None, activation_class_decoder=None, 
        bias_encoder=True, bias_decoder=True, bias_classifier=True, 
        dropout=0.0, init_scale=None, init_params=False, type_init='normal', seed=None, last_layer_zero_init=False)
    print(model)
    x = torch.randint(low=0, high=input_dim, size=(batch_size, 2)) # (batch_size, 2)
    x = torch.nn.functional.one_hot(x, num_classes=input_dim).float() # (batch_size, 2, input_dim)
    print(model)
    logits, h = model(x, activation=True)
    print(logits.shape, h.shape)