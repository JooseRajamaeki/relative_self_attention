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

        assert(not np.isnan(np.min(outputs.detach().numpy())))

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


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    random.seed(8435709)
    np.random.seed(8435709)

    sequence_length = 20
    batch_num = 7

    in_dimension = 5
    out_dimension = 3

    assert(in_dimension > out_dimension)

    x = np.zeros([batch_num,sequence_length,in_dimension])
    for batch in range(batch_num):
        value = 1.5
        step = random.randint(5,15)
        for i in range(step,sequence_length):
            x[batch,i,:] = value

    y = np.zeros([batch_num,sequence_length,out_dimension])
    for batch in range(batch_num):
        for step in range(sequence_length):
            for i in range(out_dimension):
                if step > 0:
                    y[batch,step,i] = x[batch,step,i] - x[batch,step-1,i]
                else:
                    y[batch,step,i] = x[batch,step,i]
            

    x = torch.Tensor(x)
    y = torch.Tensor(y)

    # Change the attention type to any other string if you want to use the regular self attention.
    attention_type = 'relative'

    hidden_dimension = 100
    attention = torch.nn.Sequential(SelfAttention(in_dimension,hidden_dimension,attention_type=attention_type),
                                    torch.nn.SELU(),
                                    SelfAttention(hidden_dimension,hidden_dimension, attention_type=attention_type),
                                    torch.nn.SELU(),
                                    SelfAttention(hidden_dimension,out_dimension, attention_type=attention_type))

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(attention.parameters(),lr=1e-4,momentum=0.8)

    for iteration in range(5000):

        optimizer.zero_grad()
        y_hat = attention(x)

        loss = criterion(y_hat,y)
        loss.backward()

        optimizer.step()

        print(loss.item())

    y_hat = y_hat.detach().numpy()

    for batch in range(batch_num):
        dimension = 0
        plt.plot(y_hat[batch,:,dimension])
        plt.plot(y[batch,:,dimension])
        plt.plot(x[batch,:,dimension])
        plt.legend(['Prediction','Output','Input'])
        plt.title('Dimension '+str(dimension)+' Batch '+str(batch))
        plt.show()