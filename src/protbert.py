from transformers import BertModel, BertTokenizer, BertForMaskedLM
import pandas as pd
import numpy as np
from tqdm import tqdm 
import torch
import scipy
import os
import sys
from .utils import get_pseudo_likelihood
import pickle as pkl

class ProtBert():

    """
    Class for the protein Language Model
    """

    def __init__(self, method = "average", file_name = "."):
        """
        Creates the instance of the language model instance, loads tokenizer and model

        parameters
        ----------

        method: `str`
        Which token to use to extract embeddings of the last layer
        
        file_name: `str`
        The name of the folder to store the embeddings
        """
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

        self.model = BertModel.from_pretrained("Rostlab/prot_bert").to(self.device)
    
        self.mask_model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert").to(self.device)

        self.method = method
        self.file = file_name


    def fit_transform(self, sequences:list, batches = 10):
        """
        Fits the model and outputs the embeddings.
        
        parameters
        ----------

        sequences: `list` 
        List with sequences to be transformed
        
        batches: `int`
        Number of batches. Per batch a checkpoint file will be saved

        return
        ------

        None, saved the embeddings in the embeddings.csv
        """
        batch_size = round(len(sequences)/batches)
        
        pooler_zero = np.zeros((len(sequences), 1024))
        print("\nUsing the {} method".format(self.method))
        for sequence,_ in zip(enumerate(sequences), tqdm(range(len(sequences)))):
            if not isinstance(sequence[1], float):
                #seq_tokens = ' '.join(list(sequence[1]))
                tokenized_sequences = self.tokenizer(sequence[1], return_tensors= 'pt').to(self.device) #return tensors using pytorch
                output = self.model(**tokenized_sequences)
                if self.method == "average":
                    output = torch.mean(output.last_hidden_state, axis = 1)[0]
                
                elif self.method == "pooler":
                    output = output.pooler_output[0]
                elif self.method == "first":
                    output = output.last_hidden_state[:,0][0]
                elif self.method == "last":
                    output = output.last_hidden_state[:,-1][0]
                    
                pooler_zero[sequence[0],:] = output.tolist()

        return pd.DataFrame(pooler_zero,columns=[f"dim_{i}" for i in range(pooler_zero.shape[1])])


    def calc_evo_likelihood_matrix_per_position(self, sequences: list):
        probs = []
        self.mask_model = self.mask_model.to(self.device)

        for sequence in tqdm(sequences):
            seq_tokens = ' '.join(list(sequence))
            seq_tokens = self.tokenizer(seq_tokens, return_tensors='pt')
            seq_tokens = seq_tokens.to(self.device)
            logits = self.mask_model(**seq_tokens).logits[0].cpu().detach().numpy()
            prob = scipy.special.softmax(logits,axis = 1)
            df = pd.DataFrame(prob, columns = self.tokenizer.vocab)
            df = df.iloc[:,5:-5]
            df = df.loc[:, df.columns.isin(["U","Z","O","B","X"]) == False]
            #removing CLS and SEP
            df = df.iloc[1:-1,:]
            df = df.reindex(sorted(df.columns), axis=1)
            probs.append(df)

        likelihoods = get_pseudo_likelihood(probs, sequences) 
        pkl.dump([probs,likelihoods],open("outfiles/"+self.file+"/probabilities_pseudo.pkl","wb"))

    def calc_pseudo_likelihood_sequence(self, sequences: list):
        pll_all_sequences = []
        self.mask_model = self.mask_model.to(self.device)

        for j,sequence in enumerate(tqdm(sequences)):
            try: 
                amino_acids = list(sequence)
                seq_tokens = ' '.join(amino_acids)
                seq_tokens = self.tokenizer(seq_tokens, return_tensors='pt')
                seq_tokens = seq_tokens.to(self.device)
                logits = self.mask_model(**seq_tokens).logits[0].cpu().detach().numpy()
                prob = scipy.special.softmax(logits,axis = 1)
                df = pd.DataFrame(prob, columns = self.tokenizer.vocab)
                df = df.iloc[1:-1,:]

                per_position_ll = []
                for i in range(len(amino_acids)):
                    aa_i = amino_acids[i]
                    ll_i = np.log(df.iloc[i,:][aa_i])
                    per_position_ll.append(ll_i)
                
                pll_seq = np.average(per_position_ll)
                pll_all_sequences.append(pll_seq)
            except:
                pll_all_sequences.append(None)

        return pll_all_sequences

    def calc_probability_matrix(self, sequence:str):
        amino_acids = list(sequence)
        seq_tokens = ' '.join(amino_acids)
        seq_tokens = self.tokenizer(seq_tokens, return_tensors='pt')
        seq_tokens = seq_tokens.to(self.device)
        logits = self.mask_model(**seq_tokens).logits[0].cpu().detach().numpy()
        prob = scipy.special.softmax(logits,axis = 1)
        df = pd.DataFrame(prob, columns = self.tokenizer.vocab)
        df = df.iloc[1:-1, 5:-5] # Newly added
        
        return df
