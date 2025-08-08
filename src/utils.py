import os 
import pandas as pd
import numpy as np
import re 
import warnings
import logging
from transformers import logging as transformers_logging

#Log warning messages
logging.basicConfig(level=logging.WARNING)
transformers_logging.set_verbosity_error()

def calculate_mutations(sequences_file, prob_matrix, num_mutations, seq_id_column, sequences_column):
        
    # Extract the sequence and sequence IDs from the DataFrame
    sequences = sequences_file[[seq_id_column, sequences_column]].to_dict(orient='records')
    sequence_id = sequences[0][seq_id_column]
    sequence = sequences[0][sequences_column]

    # Create a list to store mutations for the current sequence
    mutations = []
    
    # Iterate over each position in the sequence
    for position in range(min(len(sequence), len(prob_matrix))):
    
        original_token = sequence[position]
        position_probs = prob_matrix.iloc[position]

        top_tokens = position_probs.nlargest(3)

        for mutated_token, highest_prob_value in top_tokens.items():
            if mutated_token == original_token:
                continue

            original_prob = position_probs.get(original_token, 0)
            delta = highest_prob_value - original_prob

            mutations.append({
                'sequence_id': sequence_id,
                'sequence': sequence,
                'position': position + 1,
                'original_token': original_token,
                'mutated_token': mutated_token,
                'predicted_probabilities': highest_prob_value,
                'original_probabilities': original_prob,
                'delta_probabilities': delta
            })
            
    # Select the main mutations
    mutations.sort(key=lambda x: x['delta_probabilities'], reverse=True)
    selected_mutations = mutations[:int(num_mutations)]

    # Converts to DataFrame and formats with the delimiter |
    if selected_mutations:
        selected_mutations_df = pd.DataFrame(selected_mutations)
        selected_mutations_df['mutations'] = selected_mutations_df.apply(lambda row: f"{row['original_token']}{row['position']}{row['mutated_token']}", axis=1)
        
        # Groups and combines mutations in delimited formato
        grouped_mutations_df = selected_mutations_df.groupby('sequence_id').agg({
            'sequence': 'first',
            'mutations': lambda x: '|'.join(x),
            'predicted_probabilities': lambda x: '|'.join(map(str, x)),
            'original_probabilities': lambda x: '|'.join(map(str, x)),
            'delta_probabilities': lambda x: '|'.join(map(str, x))
        }).reset_index()
        
        return grouped_mutations_df
    else:
        print(f"No significant mutations found for sequence ID {sequence_id}.")
        return pd.DataFrame()