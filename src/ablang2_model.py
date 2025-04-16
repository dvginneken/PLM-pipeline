import ablang2
import pandas as pd
import scipy
import sys
import torch

sys.path.append("../scripts")

class Ablang2():

    """
    Class for the protein Model Ablang2
    """

    def __init__(self, file_name = ".", method = "seqcoding"):
        """
        Creates the instance of the language model instance; either light or heavy
        
        method: `str`
        Which token to use to extract embeddings of the last layer
        
        file_name: `str`
        The name of the folder to store the embeddings
        """
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Download and initialise the model
        self.model = ablang2.pretrained(model_to_use='ablang2-paired', device=self.device)

        #dont update the weights
        self.model.freeze()

        self.file = file_name
        self.mode = method
    


    def fit_transform(self, sequences):
        """
        Fits the model and outputs the embeddings.
        
        parameters
        ----------

        sequences: `list` 
        List with sequences to be transformed VH and VL are separated with |
        ------

        None, saved the embeddings in the embeddings.csv
        """
        all_seqs = [sequence.split("|") for sequence in sequences]  # Split sequences in VH an VL 
        output = self.model(all_seqs, mode=self.mode)
        
        if self.mode == "seqcoding":
            #The embeddings are made my averaging across all residues    
            return pd.DataFrame(output,columns=[f"dim_{i}" for i in range(output.shape[1])])
        

    def calc_pseudo_likelihood_sequence(self, sequences: list):
        """
        Calculate the pseudolikelihood of a list of sequences.
        
        parameters
        ----------

        sequences: `list` 
        List with sequences to be transformed, VH and VL are separated with |
        ------

        pll_all_sequences: `array`
        """
        all_seqs = [sequence.split("|") for sequence in sequences]  # Split sequences in VH an VL 
        pll_all_sequences = self.model(all_seqs, mode='pseudo_log_likelihood') # Calculate pseudolikelihood
        return pll_all_sequences

    def calc_probability_matrix(self, sequence:str):
        """
        Calculate the probability matrix of a sequence.
        
        parameters
        ----------

        sequence: `string` 
        Sequence to be transformed, VH and VL are separated with |
        ------

        prob_matrix: `dataframe`
        Probability matrix of each amino acid in this VH+VL sequence
        """
        sequence = [sequence.split("|")] # Split sequence in VH an VL
        logits = self.model(sequence, mode="likelihood")[0] # Calculate the likelihood
        prob = scipy.special.softmax(logits,axis = 1) # Softmax transformation
        prob_matrix = pd.DataFrame(prob, columns = list(self.model.tokenizer.decode(range(0,26)))).iloc[1:-4,] # Transform to dataframe and remove the start, stop and sep positions
        prob_matrix = prob_matrix.drop(columns=['<','>','*','X','|','-']) # Drop special tokens
        prob_matrix = prob_matrix.reindex(sorted(prob_matrix.columns), axis=1) # Sort columns on alphabetical order
        return prob_matrix