import torch
import numpy as np
import math
import random

def _affine(weight, bias, inputs):
    '''
    Inputs should have the dimensions [batch_size, sequence_length, input_dimension]
    '''

    assert(len(list(inputs.size())) == 3)

    outputs_linear = torch.einsum('ij,klj->kli', weight, inputs)
    outputs_affine = outputs_linear + bias

    return outputs_affine


class SelfAttention(torch.nn.Module):

    def __init__(self,in_dim,out_dim, attention_type = 'regular'):

        super(SelfAttention, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.attention_type = attention_type

        k = 1.0 / math.sqrt(in_dim)
        self.weights_key = torch.nn.Parameter(torch.Tensor(np.random.uniform(-k,k,size=[out_dim,in_dim])))
        self.weights_query = torch.nn.Parameter(torch.Tensor(np.random.uniform(-k,k,size=[out_dim,in_dim])))
        self.weights_value = torch.nn.Parameter(torch.Tensor(np.random.uniform(-k,k,size=[out_dim,in_dim])))

        self.bias_key = torch.nn.Parameter(torch.Tensor(np.random.uniform(-k,k,[out_dim])))
        self.bias_query = torch.nn.Parameter(torch.Tensor(np.random.uniform(-k,k,[out_dim])))
        self.bias_value = torch.nn.Parameter(torch.Tensor(np.random.uniform(-k,k,[out_dim])))

    def forward_batch(self,inputs):

        keys = _affine(self.weights_key,self.bias_key,inputs)
        queries = _affine(self.weights_query,self.bias_query,inputs)
        values = _affine(self.weights_value,self.bias_value,inputs)

        raw_weights = torch.einsum('ijk,ilk->ijl', keys, queries)
        raw_weights = raw_weights / math.sqrt(self.out_dim)

        weights = torch.softmax(raw_weights,axis=2)

        if self.attention_type == 'relative':
            weights = torch.triu(weights,diagonal=0) - torch.tril(weights,diagonal=-1)


        outputs = torch.einsum('ijk,ikl->ijl',weights,values)

        return outputs

    def forward(self,inputs):

        batch = True
        if len(list(inputs.size())) == 2:
            batch = False
            inputs = inputs[np.newaxis,:]

        outputs = self.forward_batch(inputs)

        if not batch:
            outputs = outputs[0,:,:]

        return outputs


class MultiHeadSelfAttention(torch.nn.Module):

    def __init__(self, in_dim, out_dim, head_dim, num_heads, attention_type = 'regular'):

        super(MultiHeadSelfAttention, self).__init__()

        k = 1.0 / math.sqrt(head_dim * num_heads)
        self.weights = torch.nn.Parameter(torch.Tensor(np.random.uniform(-k,k, size=[out_dim, head_dim * num_heads])))
        self.bias = torch.nn.Parameter(torch.Tensor(np.random.uniform(-k,k, size=[out_dim])))

        self.heads = []
        for i in range(num_heads):
            self.heads.append(SelfAttention(in_dim,head_dim,attention_type=attention_type))
            self.add_module('self_attention_'+str(i),self.heads[-1])
        
    def forward(self,inputs):

        batch = True
        if len(list(inputs.size())) == 2:
            batch = False
            inputs = inputs[np.newaxis,:]

        head_outputs = []
        for head in self.heads:
            head_outputs.append(head.forward_batch(inputs))

        concatenated = torch.cat(head_outputs, dim=2)

        outputs = _affine(self.weights,self.bias,concatenated)

        if not batch:
            outputs = outputs[0,:,:]

        return outputs


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    random.seed(8435709)
    np.random.seed(8435709)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    sequence_length = 20
    batch_size = 7

    in_dimension = 5
    out_dimension = 3

    assert(in_dimension > out_dimension)

    x = np.zeros([batch_size,sequence_length,in_dimension])
    for batch_idx in range(batch_size):
        value = random.uniform(0.5,2.5)
        step = random.randint(5,15)
        for i in range(step,sequence_length):
            x[batch_idx,i,:] = value

    y = np.zeros([batch_size,sequence_length,out_dimension])
    for batch_idx in range(batch_size):
        for step in range(sequence_length):
            for i in range(out_dimension):
                if step > 0:
                    y[batch_idx,step,i] = x[batch_idx,step,i] - x[batch_idx,step-1,i]
                else:
                    y[batch_idx,step,i] = x[batch_idx,step,i]
            

    x = torch.Tensor(x)
    x = x.to(device)
    y = torch.Tensor(y)
    y = y.to(device)

    # Change the attention type to any other string if you want to use the regular self attention.
    attention_type = 'relative'

    hidden_dimension = 25
    head_dim = 10
    num_heads = 5
    
    attention = torch.nn.Sequential(MultiHeadSelfAttention(in_dimension,hidden_dimension,head_dim,num_heads,attention_type=attention_type),
                                    torch.nn.SELU(),
                                    MultiHeadSelfAttention(hidden_dimension,hidden_dimension,head_dim,num_heads, attention_type=attention_type),
                                    torch.nn.SELU(),
                                    MultiHeadSelfAttention(hidden_dimension,out_dimension,head_dim,num_heads, attention_type=attention_type))

    attention = attention.to(device)

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(attention.parameters(),lr=1e-4,momentum=0.8)

    for iteration in range(3000):

        optimizer.zero_grad()
        y_hat = attention(x)

        loss = criterion(y_hat,y)
        loss.backward()

        optimizer.step()

        endline = '\r'
        if iteration % 100 == 0:
            endline = '\n'
        print('Loss: {0:.5} iteration {1}\t\t\t'.format(loss.item(),iteration),end=endline)

    y_hat = y_hat.to('cpu').detach().numpy()
    y = y.to('cpu').numpy()
    x = x.to('cpu').numpy()

    for batch_idx in range(batch_size):
        dimension = 0
        plt.plot(y_hat[batch_idx,:,dimension])
        plt.plot(y[batch_idx,:,dimension])
        plt.plot(x[batch_idx,:,dimension])
        plt.legend(['Prediction','Output','Input'])
        plt.title('Dimension '+str(dimension)+' Batch '+str(batch_idx))
        plt.show()