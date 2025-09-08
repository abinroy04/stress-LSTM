import torch.nn.functional as F
import torch
from dataclasses import dataclass
from typing import List, Union, Any, Dict


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    forced_decoder_ids: int
    eos_token_id: int
    transcription_column_name: str

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate and prepare batch for training or inference.
        """
        # Check if features list is empty
        if not features:
            return {}
            
        # Create a batch dictionary
        batch = {}
        
        # Extract and collect word boundaries
        if "word_boundaries" in features[0]:
            batch["word_boundaries"] = [f.get("word_boundaries", []) for f in features]
        
        # Extract wav2vec features
        if "wav2vec_inputs" in features[0]:
            wav2vec_inputs = [f["wav2vec_inputs"] for f in features]
            
            # Pad to the same length
            if all(isinstance(x, torch.Tensor) for x in wav2vec_inputs):
                max_length = max(x.shape[0] for x in wav2vec_inputs)
                padded_features = []
                
                for feature in wav2vec_inputs:
                    if feature.shape[0] < max_length:
                        padding = torch.zeros((max_length - feature.shape[0], *feature.shape[1:]))
                        feature = torch.cat([feature, padding], dim=0)
                    padded_features.append(feature)
                    
                batch["wav2vec_inputs"] = torch.stack(padded_features)
            else:
                # Just collect them without padding
                batch["wav2vec_inputs"] = wav2vec_inputs
        
        # Handle emphasis head labels - modify to match the model's max_words
        labels_head_key = 'labels_head'
        labels_head_key_opts = [elem for elem in list(features[0].keys()) if "labels_head" in elem and self.transcription_column_name in elem]
        
        if labels_head_key_opts:
            labels_head_key = labels_head_key_opts[0]
            
            # Extract labels_head
            labels_head = [{"labels_head": feature[labels_head_key]} for feature in features]
            
            # Get the max length across all features
            max_len = max([len(f["labels_head"]) for f in labels_head])
            
            # The model has a fixed max_words parameter (usually 20)
            # We should ensure labels are aligned with this parameter
            model_max_words = 20  # Should match CONFIG["max_words"]
            
            # Create labels that match the model's expected size
            padded_labels = []
            for f in labels_head:
                # Take up to model_max_words tokens
                feature_len = min(len(f["labels_head"]), model_max_words)
                
                # Create a tensor of -100s (ignore index for loss)
                padded = torch.full((model_max_words,), -100, dtype=f["labels_head"].dtype)
                
                # Fill in the actual labels
                padded[:feature_len] = f["labels_head"][:feature_len]
                padded_labels.append(padded)
            
            batch['labels_head'] = torch.stack(padded_labels)
        
        # Add sentence indices if available
        if "sentence_index" in features[0]:
            batch["sentence_index"] = torch.tensor([feature["sentence_index"] for feature in features])
        
        return batch