from transformers import AutoTokenizer, EsmModel, EsmForMaskedLM
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from esm import pretrained
import pickle as pkl
import os
import sys
import scipy

from .utils import get_pseudo_likelihood


class ESM1b():

    """
    Class for the protein Language Model
    """

    def __init__(self, method = "average", file_name = ".", cache_dir = "default"):
        
        """
        Creates the instance of the language model instance, loads tokenizer and model

        parameters
        ----------

        method: `str`
        Which token to use to extract embeddings of the last layer
        
        file_name: `str`
        The name of the folder to store the embeddings
        """
        

        torch.cuda.empty_cache()

        self.name_ = "esm1b_t33_650M_UR50S"
        self.method = method
        self.file = file_name
        self.repr_layer_ = -1
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Check if a cache directory is specified, otherwise use the default
        if cache_dir != "default":
            CACHE_DIR = cache_dir
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S", cache_dir=CACHE_DIR)
            self.model = EsmModel.from_pretrained("facebook/esm1b_t33_650M_UR50S", cache_dir=CACHE_DIR).to(self.device)
            self.mask_model = EsmForMaskedLM.from_pretrained("facebook/esm1b_t33_650M_UR50S", cache_dir=CACHE_DIR).to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S")
            self.model = EsmModel.from_pretrained("facebook/esm1b_t33_650M_UR50S").to(self.device)
            self.mask_model = EsmForMaskedLM.from_pretrained("facebook/esm1b_t33_650M_UR50S").to(self.device)

        

    def fit_transform(self, sequences:list, batches = 10):
        """
        Fits the model and outputs the embeddings.
        
        parameters
        ----------

        sequences: `list` 
        List with sequences to be transformed
        
        batches: `int`
        Number of batches. Per batch a checkpoint file will be saved
        ------

        None, saved the embeddings in the embeddings.csv
        """
        batch_size = round(len(sequences)/batches)
        print("\nUsing the {} method".format(self.method))
        
        pooler_zero = np.zeros((len(sequences),1280))
        for sequence,_ in zip(enumerate(sequences), tqdm(range(len(sequences)))):
            if not isinstance(sequence[1], float):
                j = sequence[0]
                amino_acids = list(sequence[1])
                seq_tokens = ' '.join(amino_acids)
                tokenized_sequences = self.tokenizer(seq_tokens, return_tensors= 'pt') #return tensors using pytorch
                tokenized_sequences = tokenized_sequences.to(self.device)
                output = self.model(**tokenized_sequences)

                if self.method == "average":
                    output = torch.mean(output.last_hidden_state, axis = 1)[0]
                
                elif self.method == "pooler":
                    output = output.pooler_output[0]
                
                elif self.method == "last":
                    output = output.last_hidden_state[0,-1,:]

                elif self.method == "first":
                    output = output.last_hidden_state[0,0,:]
                    
                pooler_zero[sequence[0],:] = output.tolist()

        return pd.DataFrame(pooler_zero,columns=[f"dim_{i}" for i in range(pooler_zero.shape[1])])

    def calc_evo_likelihood_matrix_per_position(self, sequences:list, batch_size = 10):

        batch_converter = self.alphabet_.get_batch_converter()
        data = []
        for i,sequence in enumerate(sequences):
            data.append(("protein{}".format(i),sequence))
        probs = []
        count = 0
        #One sequence at a time
        for sequence,_ in zip(data,tqdm(range(len(data)))):
            #Tokenize & run using the last layer
            _, _, batch_tokens = batch_converter([sequence])
            batch_tokens = batch_tokens.to("cuda:0" if torch.cuda.is_available() else "cpu")
            out = self.model_(batch_tokens,repr_layers = [self.repr_layer_],return_contacts = False)
            #Retrieve numerical values for each possible token (including aminoacids and special tokens) in each position
            logits = out["logits"][0].cpu().detach().numpy()
            #Turn them into probabilties 
            prob = scipy.special.softmax(logits,axis = 1)
            #Preprocessing probabilities, removing CLS and SEP tokens and removing probabilities of Special aminoacids and tokens of the model.
            df = pd.DataFrame(prob, columns = self.alphabet_.all_toks)
            df = df.iloc[:,4:-4]
            df = df.loc[:, df.columns.isin(["U","Z","O","B","X"]) == False]
            #removing CLS and SEP
            df = df.iloc[1:-1,:]
            df = df.reindex(sorted(df.columns), axis=1)
            probs.append(df)

            count+=1

        likelihoods = get_pseudo_likelihood(probs, sequences)
        pkl.dump([probs,likelihoods],open("outfiles/"+self.file+"/probabilities_pseudo.pkl","wb"))
        print("done with predictions")

        return(probs)

    def calc_pseudo_likelihood_sequence(self, sequences:list):

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
                df = pd.DataFrame(prob, columns = self.tokenizer.convert_ids_to_tokens(range(0,33)))
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
    
    def calc_probability_matrix(self,sequence:str):

        amino_acids = list(sequence)
        seq_tokens = ' '.join(amino_acids)
        seq_tokens = self.tokenizer(seq_tokens, return_tensors='pt')
        seq_tokens = seq_tokens.to(self.device)
        logits = self.mask_model(**seq_tokens).logits[0].cpu().detach().numpy()
        prob = scipy.special.softmax(logits,axis = 1)
        df = pd.DataFrame(prob, columns = self.tokenizer.convert_ids_to_tokens(range(0,33)))
        df = df.iloc[1:-1, 4:-9] # Newly added
        
        return df
