import torch
import torch.randn as randn
import torch.nn as nn
import torch.squeeze as squeeze
from torch.autograd import Variable
import numpy as np

class MemN2N(nn.Module):
    def __init__(self, batch_size, memory_size, vocab_size, embed_size, hops):
        super(MemN2N, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.hops = hops
        #Adjacent memory
        #the output embedding for one layer is the input embedding for the one above [Sukhbaatar et al.]
        #A1, C1 == A2, C2 == A3, ..., Actually hops + 1 embeddings are needed
        self.memory = []
        A = nn.Embedding(self.vocab_size, self.embed_size) #input memory
        A.weight = nn.Parameter(randn(self.vocab_size, self.embed_size).normal_(0, 0.1))
        for i in range(self.hops):
            p = nn.Softmax() #softmax layer between input and output
            C = nn.Embedding(self.vocab_size, self.embed_size) #output memory
            C.weight = nn.Parameter(randn(self.vocab_size, self.embed_size).normal_(0, 0.1))
            self.memory.append([A, p, C])
            A = C #A[i+1] = C[i]
        # final weight matrix
        self.W = nn.Parameter(randn(self.embed_size, self.vocab_size), requires_grad=True)
        # final softmax layer
        self.m = nn.Softmax()

    def get_position_encoding(self, query_size):
        """
        Position Encoding (PE)
        the order of the words now affects.
        """
        encoding = np.ones((self.embed_size, query_size), dtype=np.float32)
        ls = query_size + 1
        le = self.embed_size + 1
        for i in range(1, le):
            for j in range(1, ls):
                encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
        encoding = 1 + 4 * encoding / self.embed_size / query_size
        enc_vec = torch.from_numpy(np.transpose(encoding)).type(torch.FloatTensor)
        return enc_vec

    def embed_evidences(self, evidences, embedding_layer):
        evidence_embedding_list = []
        for evidence in evidences:
            evidence_variable = Variable(torch.squeeze(evidence, 0).data.type(torch.LongTensor))
            evidence_embedding = embedding_layer(evidence_variable)
            position_encoding = self.get_position_encoding(evidence.size(), self.embed_size)
            evidence_embedding = evidence_embedding * position_encoding
            evidence_embedding_list.append(evidence_embedding)

        batch_story_embedding_temp = torch.stack(evidence_embedding_list)
        batch_story_embedding = torch.sum(batch_story_embedding_temp, dim=2)
        return torch.squeeze(batch_story_embedding, dim=2)

    def forward(self, x_e, x_q):
        e = Variable(x_e, requires_grad=False) #evidences
        q = Variable(squeeze(x_q, 1), requires_grad=False) #question
        u_list = []
        #emb query
        queries_emb = self.memory[0][0](q) #in the simplest case via another embedding matrix B with the same dimensions as A
        position_encoding = self.get_position_encoding(queries_emb.size()[0])
        queries = queries_emb * position_encoding
        u_list.append(torch.sum(queries, dim=1))
        for i in range(self.hops):
            #emb A
            evidence_emb_A = self.embed_story(e, self.memory[i][0])
            #inner product
            u_k_1_matrix = [u_list[-1]] * self.memory_size
            p = evidence_emb_A * torch.squeeze(torch.stack(u_k_1_matrix, dim=1), 2)
            #softmax
            p = self.memory[i][1](torch.squeeze(torch.sum(p, dim=2)))
            #emb_C
            evidence_emb_C = self.embed_story(e, self.memory[i][2])
            #inner product
            pre_o = torch.mul(evidence_emb_C, p.unsqueeze(1).expand_as(evidence_emb_C))
            o = torch.sum(pre_o, dim=2)
            #u_k
            u_list.append(torch.squeeze(o) + torch.squeeze(u_list[-1]))
        wx = torch.mm(u_list[-1], self.W)
        y_pred = self.m(wx)
        return y_pred



#Test
batch_size = 1
embed_size = 4
vocab_size = 4
hops = 2
memory_size = 10
net = MemN2N(batch_size, memory_size, vocab_size, embed_size, vocab_size, hops)