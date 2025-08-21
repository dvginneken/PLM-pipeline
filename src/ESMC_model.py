
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import scipy
import torch
import numpy as np
import sys
import pandas as pd
import os

sys.path.append("../scripts")


class ESMc():

    """
    Class for the protein Language Model ESMC
    """

    def __init__(self, ):
        
        """
        Creates the instance of the language model and loads model

        """
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = ESMC.from_pretrained("esmc_600m").to(self.device)

        

    def fit_transform(self, sequence_file, layer:str = "last", method:str = "average_pooling", save_path:str = ".", model_name:str = "ESMc", 
                      seq_id_column:str = "sequence_id", sequences_column:str = "sequence"):
        """
        Fits the model and outputs the embeddings.
        
        parameters
        ----------

        sequence_file: `dataframe` 
        DataFrame with sequences to be transformed
        
        layer: `str`
        Layer from which to extract the embeddings. Default is "last".

        method: `str`
        Method to extract the embeddings. Default is "average_pooling". Options: "average_pooling", "per_token".

        save_path: `str`
        Path to save the embeddings CSV file. Default is current directory.

        model_name: `str`
        Name of the model, used for saving the embeddings file. Default is "ESMc".

        seq_id_column: `str`
        Column name in the sequence_file DataFrame that contains unique sequence identifiers. Default is "sequence_id".

        sequences_column: `str`
        Column name in the sequence_file DataFrame that contains sequences. Default is "sequence".
        ------

        None, saved the embeddings in the embeddings.csv
        """

        print("\nUsing the {} method".format(method))
        
        pooler_zero = np.zeros((len(sequence_file.index),1152))
        for index, row in sequence_file.iterrows():
            sequence = row[sequences_column]
            seq_id = row[seq_id_column]
            protein = ESMProtein(sequence=sequence)
            protein_tensor = self.model.encode(protein) # Tokenize the sequence
            if layer == "last":
                embeddings_output = self.model.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)).embeddings[0] # Get the embeddings of the last hidden layer
            #TO DO
            # if layer == ..
            #    
            if method == "average_pooling": # Average over all residues for each head
                output = torch.mean(embeddings_output, axis = 0)
                pooler_zero[index,:] = output.tolist()

            elif method == "per_token": # Per token embeddings
                output = embeddings_output
                embeds = pd.DataFrame(output.cpu().numpy(), columns=[f"dim_{i}" for i in range(output.shape[1])])
                embeds = embeds.iloc[1:-1,:] # Remove start and stop token
                embeds.to_csv(os.path.join(save_path,f"embeddings_seq_{seq_id}_{model_name}.csv"), index = False)

        # Save the average embeddings to a CSV file
        if method == "average_pooling":
            embeds = pd.DataFrame(pooler_zero,columns=[f"dim_{i}" for i in range(pooler_zero.shape[1])])
            embeds = pd.concat([sequence_file,embeds],axis=1) # Add to the sequence file 
            embeds.to_csv(os.path.join(save_path,f"embeddings_{model_name}.csv"), index=False)



    def calc_pseudo_likelihood_sequence(self, sequences:list):
        """
        Calculates the pseudolikelihood of a list of sequences.
        
        parameters
        ----------
        sequences: `list` 
        List with sequences to be transformed

        returns
        -------
        pll_all_sequences: `list`
        List of pseudolikelihood values for each sequence.
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

        returns
        -------
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
        return prob_matrix

    def calc_attention_matrix(self, sequence:str, layer:str = "last", head:str = "average"):
        """
        Calculates the attention matrix for a given sequence and layer.

        parameters
        ----------
        sequence: `str`
        The input protein sequence.

        layer: `int`
        The layer from which to extract the attention scores. Default is -1 (last layer).

        head: `str`
        The attention head to extract scores from. Default is "average".

        returns
        -------
        attn_matrix: `DataFrame`
        A DataFrame containing the attention matrix for the sequence.s
        """
        print("Attention matrices for ESMC are not implemented yet.")
        
        df = pd.DataFrame()

        return df