import torch.nn as nn
import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

## Inputs transformed 
## Decoder from the original translation mechanism

# The last output of the original encoder is sometimes called the context vector as it encodes context from the entire sequence. This context vector is used as the initial hidden state of the decoder.

# The initial hidden state of our decoder is the output of the FF network.

# At every step of decoding, the decoder is given an input token and hidden state. The initial input token is the start-of-string <SOS> token, and the first hidden state is the encoder’s output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def input_data(in_file):
	"""Extract inputs from csv file with 13C shifts and smiles returns X transformed, smiles list, max smiles length, smiles character dictionary """
	sss = ""
	x = []
	ml = 0 # max length
	smis = []
	ind2w = {0: "SOS", 1: "EOS"}
	d = {}
	lets =2 # sos and eos 
	for line in in_file:
		
		spec = (line.split("],")[0]).split("[")[1].split(",")
		x.append(spec)
		s = line.split("],")[1].strip()
		for letter in s:
			if letter not in d:
				d[letter] = int(lets)
				ind2w[lets] = letter
				lets += 1 
		smis.append(s)
		
		if len(s) > ml:
			sss = s
			ml = len(s)
		else:
			sss = sss
			ml = ml
	print("max length:", ml, "smiles:", sss) 
	i = [torch.Tensor([float(j) for j in i]) for i in x]
	X_in = pad_sequence(i,batch_first=True)
	quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal',random_state=0)
	X_trans = quantile_transformer.fit_transform(np.array(X_in))		
	X_trans = torch.from_numpy(X_trans)
	vocab_size = len(ind2w)
	return X_trans, smis, ml, d, ind2w, vocab_size
f = open('nmr-smi.txt','r')

X_trans, smiles, ml, dic,ind2w,vocab_size  = input_data(f)
print(dic)
unit = len(ind2w)


def split(X,smis,v):
	"""generates list of pairs list"""
	pair=[]
	for k in range(len(X)):
		pair.append(list((list(X[k]),smis[k])))
	random.seed(32)
	random.shuffle(pair)
	splen = round(len(pair)/v)
	tt =[]
	for k in range(v):
		uu =[]
		for i in range(k*splen,(k+1)*splen):
			if i == len(pair):
				break
			uu.append(pair[i])
		tt.append(uu)

	
	return tt, pair

v=4 # sfold

hlist,pair= split(X_trans,smiles,v) # sfold huge list of pairs

def tensor_from_smiles(smiles_b):
	indexes = [dic[let] for let in smiles_b]
	indexes.append(int(1)) # 0 : EOS token
	in_smiles = torch.tensor(indexes,dtype=torch.long, device=device)
		
	return in_smiles


class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		
		self.f1 = nn.Linear(10,500) #input to hidden layer 
		self.l1 = nn.Tanh()
		self.f2 = nn.Linear(500,2*unit)# hidden to output layer
	def forward(self,x):
		x = self.f1(x)	
		x = self.l1(x)
		x = self.f2(x)
		return x


embed_size = 2*unit ### same as final. The feature vector is linearly transformed to have the same dimension as the input dimension of the GRU network. 
hidden_size = 2*unit
num_layers = 1 # GRU 

criterion = nn.NLLLoss()

class DecoderRNN(nn.Module):
        def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_length=ml):
                super(DecoderRNN, self).__init__()
                self.embed = nn.Embedding(vocab_size, embed_size)
                self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
                self.out = nn.Linear(hidden_size, vocab_size)
                self.softmax = nn.LogSoftmax(dim=1)
	
        def forward(self, in_smiles ,hidden):#
                embeddings = self.embed(in_smiles).view(1,1,-1)
                embeddings = F.relu(embeddings)
                hidden = hidden.unsqueeze(1).unsqueeze(1).reshape(1,1,-1)
                output, hidden = self.gru(embeddings,hidden)
                outputs = self.softmax(self.out(output[0]))
                return outputs, hidden

l_r = 0.0001
def evaluate(encoder,decoder,X,smi):

	with torch.no_grad():
		loss2 = 0
		hidden = encoder(X)
		pretarget = tensor_from_smiles(smi)
		target = torch.tensor([[0]],device=device)#0: SOS

		for di in range(len(pretarget)):
			output, hidden = decoder(target,hidden)			
			target = pretarget[di].unsqueeze(0)
			loss2 += criterion(output, target)
		voss = loss2.item()/len(pretarget)
		return  voss

def evaval(encoder,decoder,val_pairs):
	voss = 0
	for i in range(len(val_pairs)):
		X = torch.tensor(val_pairs[i][0])
		smi = val_pairs[i][1]
		voss = += evaluate(encoder,decoder,X,smi)
	voss_prom = voss/(len(val_pairs))
	return voss_prom
def evaluateR(encoder,decoder,X,max_length=ml): # ml no deberia saberse para validation set

        with torch.no_grad():
                hidden = encoder(X)
                target = torch.tensor([[0]],device=device)#0: SOS
                decoded_words = []
                for di in range(max_length):
                        output, hidden = decoder(target,hidden)
                        topv,topi = output.data.topk(1)
                        if topi.item() == int(1): # EOS 
                                decoded_words.append('<EOS>')
                                break
                        else:
                                decoded_words.append(ind2w[topi.item()])
                        target = topi.squeeze().detach()
                return decoded_words

def evaluateRandomly(encoder,decoder,ppair,n=50):
        for i in range(n):
                #choice = random.randint(0,b_size-1)
                choice = random.randint(0,len(ppair)-1)
                X = torch.tensor(ppair[choice][0])
                smi = ppair[choice][1]
		
                output_words = evaluateR(encoder,decoder,X)
                output_s = ''.join(output_words)
                print("pred:",output_s)
                print("real:",smi)
## train
epochs = 50 
for k in range(len(hlist)):
	decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, max_length=ml).to(device)
	encoder = Net().to(device)
	encoder_optimizer = optim.RMSprop(encoder.parameters(),lr = l_r,momentum=0.1)
	decoder_optimizer = optim.RMSprop(decoder.parameters(),lr = l_r,momentum=0.1)
	px = [] 
	ppx = [] 
	py=[]
	ppy=[]
	j = 0# iterations
	jj = 0# iterations
	val_pairs = hlist[k]
	x_tpair = [x for x in pair if x not in val_pairs]
	b_size = len(x_tpair)
	for i in range(epochs):
		p = 0 # for plotting
		for b in range(b_size):   # batch
			j += 1 
			p += 1
			decoder_optimizer.zero_grad()
			encoder_optimizer.zero_grad()
			loss = 0
			smiles = x_tpair[b][1]
			X_train = torch.tensor(x_tpair[b][0],device=device)
			pretarget = tensor_from_smiles(smiles)
			hidden=encoder(X_train) # inithidden , features from encoder output
			target = torch.tensor([[0]],device=device)#0: SOS
			for s in range(len(pretarget)):
				output,hidden  = decoder(target,hidden)
				target = pretarget[s].unsqueeze(0)
				loss += criterion(output, target)
			loss.backward()
			encoder_optimizer.step() 
			decoder_optimizer.step() 
			if p % 200 == 0: # plot every
				px.append(j)
				py.append(loss.item()/len(pretarget))
		print("val set evaluation, epoch:",i+1)
		evaluateRandomly(encoder,decoder,val_pairs)
		voss = evaval(encoder,decoder,val_pairs)
#		print(j,voss)
		ppx.append(j)
		ppy.append(voss)
	plt.plot(px,py)
	plt.plot(ppx,ppy,marker='P')
	plt.xlabel('iterations')
	plt.ylabel('loss')
	plt.savefig('CV'+str(k+1)+'_'+'500_'+str(hidden_size)+'embhidsize_epchs_'+str(epochs)+'.pdf',format='pdf')
	plt.clf()
	plt.cla()
	plt.close()
	print("train set eval")
	evaluateRandomly(encoder,decoder,x_tpair, n=50)


