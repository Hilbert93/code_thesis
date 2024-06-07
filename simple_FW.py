import torch
from torch import nn
from torch.nn import LayerNorm
import torch.nn.functional as fun
import numpy as np

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()

class FWMRNN(nn.Module):
    def __init__(self, isize, hsize, k_size, v_size, withFWM=True):
        super().__init__()
        q_size      = k_size  # query vector has same dimension as key vector
        self.rnn    = nn.LSTM(isize, hsize, 1, dropout=0).to(device)

        if withFWM:
          self.fwm    = FWM(hsize, k_size, v_size, q_size)
          self.linear = nn.Linear(v_size, hsize).to(device)
        else:
          # record output and hidden states of LSTM
          self.m_trace = {'out':[], 'rnn_out':[], 'rnn_cell':[]}

        self.isize  = isize
        self.hsize  = hsize
        self.hasFWM = withFWM

    def reset(self):
        pass
    
    def clear_trace(self):
        if self.hasFWM:
           self.fwm.clear_trace()
        else:
          for key in list(self.m_trace.keys()):
            del self.m_trace[key][:]

    def forward(self, inputs, hidden):
        if self.hasFWM:
          lstm_hidden, F = hidden
        else:
          lstm_hidden    = hidden

        h, lstm_hidden   = self.rnn(inputs, lstm_hidden)
        outputs = []
        if self.hasFWM:
          self.fwm.m_trace['rnn_out'].append(lstm_hidden[0].detach().cpu().numpy())
          self.fwm.m_trace['rnn_cell'].append(lstm_hidden[1].detach().cpu().numpy())
          for t, h_t in enumerate(h):
            F = self.fwm.write(h_t, F) # weights FWM
            o_t = self.fwm(h_t, F)     # output of FWM [batch_size, q_size]
            outputs.append(o_t)
          s      = torch.stack(outputs, dim=0)
          output = torch.cat((h, self.linear(s)), 2)
        else:
          output = h
          self.m_trace['out'].append(output.detach().cpu().numpy())
          self.m_trace['rnn_out'].append(lstm_hidden[0].detach().cpu().numpy())
          self.m_trace['rnn_cell'].append(lstm_hidden[1].detach().cpu().numpy())

        if self.hasFWM:
          hidden = (lstm_hidden, F)
          return output, hidden, self.fwm.m_trace
        else:
          hidden = lstm_hidden
          return output, hidden, self.m_trace

class FWM(nn.Module):
    def __init__(self, hidden_size, k_size, v_size, q_size):
        super().__init__()
        self.hidden_size = hidden_size # LSTM hidden size
        self.k_size      = k_size      # key size
        self.v_size      = v_size      # value size
        self.q_size      = q_size      # query size
        self.b_size      = 1           # writing strength size
        self.m_size      = 1           # reading strength size

        # writing weights
        self.W_k = nn.Linear(hidden_size, self.k_size).to(device)
        self.W_v = nn.Linear(hidden_size, self.v_size).to(device)
        self.W_b = nn.Linear(hidden_size, self.b_size).to(device)
        # read weights
        self.W_q = nn.Linear(hidden_size, self.q_size).to(device)
        self.W_m = nn.Linear(hidden_size, self.m_size).to(device)

        self.ln_read = LayerNorm(self.q_size, elementwise_affine=False)
        # initialize weights
        self.reset_parameters()

        # record the writing key, value, old value, writing strength, query, output reading strength,
        # FWM weights LSTM output and LSTM cell states, for each time step
        self.m_trace = {'key':[], 'value':[], 'value_old':[], 'w_beta':[], 'query':[],
                        'out':[], 'r_beta':[], 'F':[], 'rnn_out':[], 'rnn_cell':[]}
        
    def clear_trace(self):
        for key in list(self.m_trace.keys()):
            del self.m_trace[key][:]

    def reset_parameters(self):
        nn.init.xavier_normal_(self.W_k.weight)
        nn.init.xavier_normal_(self.W_v.weight)
        nn.init.xavier_normal_(self.W_q.weight)
        nn.init.xavier_normal_(self.W_b.weight)
        nn.init.xavier_normal_(self.W_m.weight)

    def write(self, z, F):
        # z: [batch_size, hidden_size]
        # F: [batch_size, k_size, q_size]

        ks = fun.elu(self.W_k(z), 1.0, False)+1 # activation function for key vector
        ks = ks / ks.sum(-1, keepdim=True)      # normalize k, crucial for stable training.
        vs = self.W_v(z)                        # new value vector
        bs = torch.sigmoid(self.W_b(z))
        vs_old = torch.einsum('bij, bj->bi', F, ks) # old value vector
        new_v  = bs * (vs - vs_old)                 # equation merging remove and insert.

        # book keeping
        self.m_trace['key'].append(ks.detach().cpu().numpy())
        self.m_trace['value'].append(vs.detach().cpu().numpy())
        self.m_trace['w_beta'].append(bs.detach().cpu().numpy())
        self.m_trace['value_old'].append(vs_old.detach().cpu().numpy())

        weight_updates = torch.einsum('bi, bj->bij', new_v, ks)
        F = F + weight_updates
        self.m_trace['F'].append(F.detach().cpu().numpy())

        return F

    def forward(self, z, F):
        qs = fun.elu(self.W_q(z), 1.0, False)+1 # activation function for query vector
        qs = qs / qs.sum(-1, keepdim=True)      # normalize q, crucial for stable training
        ms = torch.sigmoid(self.W_m(z))

        outputs = torch.einsum('bij, bj->bi', F, ms*qs)

        # book-keeping
        self.m_trace['query'].append(qs.detach().cpu().numpy())
        self.m_trace['r_beta'].append(ms.detach().cpu().numpy())
        self.m_trace['out'].append(outputs.detach().cpu().numpy())

        return outputs