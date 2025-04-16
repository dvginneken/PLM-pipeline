
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import scipy
import torch
import numpy as np
import sys
import pandas as pd

sys.path.append("../scripts")


class ESMc():

    """
    Class for the protein Language Model ESMC
    """

    def __init__(self, method = "average", cache_dir = "default"):
        
        """
        Creates the instance of the language model instance, loads tokenizer and model

        parameters
        ----------

        method: `str`
        Which token to use to extract embeddings of the last layer
        
        file_name: `str`
        The name of the folder to store the embeddings
        """
        

        self.method = method

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = ESMC.from_pretrained("esmc_300m").to(self.device)

        

    def fit_transform(self, sequences:list):
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

        print("\nUsing the {} method".format(self.method))
        
        pooler_zero = np.zeros((len(sequences),960))
        for sequence,_ in zip(enumerate(sequences), range(len(sequences))):
            protein = ESMProtein(sequence=sequence[1])
            protein_tensor = self.model.encode(protein) # Tokenize the sequence
            embeddings_output = self.model.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)).embeddings[0] # Get the embeddings of the last hidden layer   
            if self.method == "average": # Average over all residues for each head
                output = torch.mean(embeddings_output, axis = 0)
            pooler_zero[sequence[0],:] = output.tolist()

        return pd.DataFrame(pooler_zero,columns=[f"dim_{i}" for i in range(pooler_zero.shape[1])])


    def calc_pseudo_likelihood_sequence(self, sequences:list):
        """
        Calculates the pseudolikelihood of a list of sequences.
        
        parameters
        ----------

        sequences: `list` 
        List with sequences to be transformed
        ------

        pll_all_sequences: `array`
        """

        pll_all_sequences = []
        for sequence in sequences:
            try:
                protein = ESMProtein(sequence=sequence)
                protein_tensor = self.model.encode(protein) # Tokenize the sequence
                logits_output = self.model.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=False)).logits # Get the logits
                tensor = logits_output.sequence
                logits = tensor.to(dtype=torch.float32).cpu().numpy()
                prob = scipy.special.softmax(logits[0],axis = 1) # Softmax transformation
                df = pd.DataFrame(prob, columns = self.model.tokenizer.convert_ids_to_tokens(range(0,64)))
                df = df.iloc[1:-1,:] # Remove start and stop token

                # Calculate the log likelihood for each position  
                per_position_ll = []
                amino_acids = list(sequence)
                for i in range(len(amino_acids)):
                    aa_i = amino_acids[i]
                    ll_i = np.log(df.iloc[i,:][aa_i]) 
                    per_position_ll.append(ll_i)

                pll_seq = np.average(per_position_ll) # Average log likelihood over the sequence length
                pll_all_sequences.append(pll_seq)
            except:
                pll_all_sequences.append(None)

        return pll_all_sequences
    
    def calc_probability_matrix(self,sequence:str):
        """
        Calculate the probability matrix of a sequence.
        
        parameters
        ----------

        sequence: `string` 
        Sequence to be transformed, VH and VL are separated with |
        ------

        prob_matrix: `dataframe`
        Probability matrix of each amino acid in this sequence
        """
        protein = ESMProtein(sequence=sequence)
        protein_tensor = self.model.encode(protein) # Tokenize the sequence
        logits_output = self.model.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=False)).logits # Get the logits
        tensor = logits_output.sequence
        logits = tensor.to(dtype=torch.float32).cpu().numpy()
        prob = scipy.special.softmax(logits[0],axis = 1) # Softmax transformation
        df = pd.DataFrame(prob, columns = self.model.tokenizer.convert_ids_to_tokens(range(0,64)))
        prob_matrix = df.iloc[1:-1,:] # Remove start and stop token
        prob_matrix = prob_matrix.drop(columns=['<cls>','<pad>','<eos>','<unk>','.','-','|','<mask>','X',None]) # Drop special tokens
        prob_matrix = prob_matrix.reindex(sorted(prob_matrix.columns), axis=1) # Sort columns on alphabetical order
        return df
