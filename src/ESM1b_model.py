from transformers import AutoTokenizer, EsmModel, EsmForMaskedLM
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
#from esm import pretrained
import pickle as pkl
import os
import sys
import scipy

sys.path.append("../scripts")


class ESM1b():

    """
    Class for the protein Language Model
    """

    def __init__(self, cache_dir = "default"):
        
        """
        Creates the instance of the language model instance, loads tokenizer and model

        parameters
        ----------

        cache_dir: `str`
        Directory to cache the model and tokenizer
        """
        

        torch.cuda.empty_cache()

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
        if layer != "last":
            raise NotImplementedError("Only 'last' layer extraction is implemented for ESM1b model.")
        
        print("\nUsing the {} method".format(method))
        
        pooler_zero = np.zeros((len(sequence_file.index),1280))

        for index, row in sequence_file.iterrows():
            sequence = row[sequences_column]
            seq_id = row[seq_id_column]
            seq_tokens = ' '.join(list(sequence))
            protein_tensor = self.tokenizer(seq_tokens, return_tensors= 'pt') #return tensors using pytorch
            protein_tensor = protein_tensor.to(self.device)
            output = self.model(**protein_tensor)

            if layer == "last":
                embeddings_output = output.last_hidden_state[0] # Get the embeddings of the last hidden layer
            #TO DO
            # if layer == ..
            #    
            if method == "average_pooling": # Average over all residues for each head
                output = torch.mean(embeddings_output, axis = 0)
                pooler_zero[index,:] = output.tolist()

            elif method == "per_token": # Per token embeddings
                output = embeddings_output
                embeds = pd.DataFrame(output.cpu().detach().numpy(), columns=[f"dim_{i}" for i in range(output.shape[1])])
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
        List of sequences to calculate the pseudolikelihood for.

        returns
        -------
        pll_all_sequences: `list`
        List of pseudolikelihood values for each sequence.
        """

        pll_all_sequences = []
        self.mask_model = self.mask_model.to(self.device)

        for sequence in tqdm(sequences):
            if len(sequence) < 1023: #ESM1b max sequence length is 1023
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
            else:
                pll_all_sequences.append(None)

        return pll_all_sequences
    
    def calc_probability_matrix(self,sequence:str):
        """
        Calculates the probability matrix for a given sequence.

        parameters
        ----------
        sequence: `str`
        The input protein sequence.

        returns
        -------
        prob_matrix: `DataFrame`
        A DataFrame containing the probability matrix for the sequence.
        """
        amino_acids = list(sequence)
        seq_tokens = ' '.join(amino_acids)
        seq_tokens = self.tokenizer(seq_tokens, return_tensors='pt')
        seq_tokens = seq_tokens.to(self.device)
        logits = self.mask_model(**seq_tokens).logits[0].cpu().detach().numpy()
        prob = scipy.special.softmax(logits,axis = 1)
        df = pd.DataFrame(prob, columns = self.tokenizer.convert_ids_to_tokens(range(0,33)))
        df = df.iloc[1:-1, 4:-9]
        
        return df
    
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
        The attention head to extract scores from ("average" or "sum"). Default is "average".

        returns
        -------
        attn_matrix: `DataFrame`
        A DataFrame containing the attention matrix for the sequence.
        """
        amino_acids = list(sequence)
        seq_tokens = ' '.join(amino_acids)
        seq_tokens = self.tokenizer(seq_tokens, return_tensors='pt')
        seq_tokens = seq_tokens.to(self.device)
        outputs = self.mask_model(**seq_tokens, output_attentions=True)
        attn_scores = torch.stack(outputs.attentions)

        #Select the layer
        if layer == "last":
            attn_scores = attn_scores[-1][0] #final layer, batch 1
        elif layer == "last_five":
            attn_scores = attn_scores[-5:].mean(dim=0)[0] #average of last 5 layers, batch 1
            
        #Select the head(s)
        if head == "average":
            attn_matrix = attn_scores.mean(dim=0)
        elif head == "sum":
            attn_matrix = attn_scores.sum(dim=0)

        #Transform to dataframe and remove CLS and SEP tokens
        df = pd.DataFrame(attn_matrix.cpu().detach().numpy()).iloc[1:-1, 1:-1]

        return df
