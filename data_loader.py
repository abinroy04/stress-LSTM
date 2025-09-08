from datasets import load_from_disk, Sequence
from processor import DSProcessor
import os
import torch

"""
# Set Hugging Face cache directories
os.environ["HF_HOME"] = "/sd1/jhansi/interns/abin/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/sd1/jhansi/interns/abin/hf_cache/transformers"
os.environ["HF_DATASETS_CACHE"] = "/sd1/jhansi/interns/abin/hf_cache/datasets"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/sd1/jhansi/interns/abin/hf_cache/hub"
"""

def add_wav2vec_features(dataset, model):
    """
    Add wav2vec features to a dataset.
    
    Args:
        dataset: Dataset to add wav2vec features to
        model: Model containing the wav2vec processor
        
    Returns:
        Dataset with wav2vec features added
    """
    def process_audio(example):
        # MANDATORY: Validate model has wav2vec processor
        if not hasattr(model, 'wav2vec_processor'):
            raise ValueError(
                f"❌ CRITICAL ERROR: Model doesn't have wav2vec_processor! "
                f"Cannot add wav2vec features. This model REQUIRES wav2vec components!"
            )
        
        # Get audio array
        audio_array = example["audio"]["array"]
        
        # MANDATORY: Validate audio data
        if audio_array is None or len(audio_array) == 0:
            raise ValueError(
                f"❌ CRITICAL ERROR: Empty or None audio array found! "
                f"Cannot process wav2vec features for sample. No fallback allowed!"
            )
        
        # Resample audio if needed
        if example["audio"]["sampling_rate"] != 16000:
            from scipy.signal import resample
            num_samples = int(len(audio_array) * 16000 / example["audio"]["sampling_rate"])
            audio_array = resample(audio_array, num_samples)
        
        # MANDATORY: Validate resampled audio
        if len(audio_array) == 0:
            raise ValueError(
                f"❌ CRITICAL ERROR: Audio resampling resulted in empty array! "
                f"Original length: {len(example['audio']['array'])}, "
                f"Original SR: {example['audio']['sampling_rate']}. No fallback allowed!"
            )
        
        # Process with wav2vec processor
        try:
            wav2vec_inputs = model.wav2vec_processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_values.squeeze(0)
        except Exception as e:
            raise ValueError(
                f"❌ CRITICAL ERROR: wav2vec processor failed! "
                f"Error: {str(e)}. Cannot process wav2vec features. No fallback allowed!"
            )
        
        # MANDATORY: Validate wav2vec output
        if wav2vec_inputs is None or wav2vec_inputs.shape[0] == 0:
            raise ValueError(
                f"❌ CRITICAL ERROR: wav2vec processor returned empty/None output! "
                f"Output shape: {wav2vec_inputs.shape if wav2vec_inputs is not None else 'None'}. "
                f"No fallback allowed!"
            )
        
        example["wav2vec_inputs"] = wav2vec_inputs
        return example
    
    # MANDATORY: Validate dataset
    if dataset is None or len(dataset) == 0:
        raise ValueError(
            f"❌ CRITICAL ERROR: Empty or None dataset provided! "
            f"Cannot add wav2vec features. No fallback allowed!"
        )
    
    print(f"Adding wav2vec features to dataset with {len(dataset)} examples...")
    processed_dataset = dataset.map(
        process_audio,
        desc="Adding wav2vec features",
        num_proc=1  # Process one at a time to avoid GPU memory issues
    )
    
    # MANDATORY: Validate processing results
    if processed_dataset is None or len(processed_dataset) == 0:
        raise ValueError(
            f"❌ CRITICAL ERROR: wav2vec feature processing resulted in empty dataset! "
            f"Original dataset size: {len(dataset)}. No fallback allowed!"
        )
    
    # MANDATORY: Validate all samples have wav2vec features
    if "wav2vec_inputs" not in processed_dataset.column_names:
        raise ValueError(
            f"❌ CRITICAL ERROR: wav2vec_inputs column not found after processing! "
            f"Available columns: {processed_dataset.column_names}. No fallback allowed!"
        )
    
    print("✅ wav2vec features added successfully to all samples.")
    return processed_dataset


class PreprocessedDataLoader():
    """
    Generic data loading class for speech emphasis detection datasets.
    
    Handles dataset preprocessing, loading from disk or HuggingFace,
    adding necessary column indices, and preparing datasets for training.
    
    Attributes:
        preprocessed_dataset_path: Root directory for preprocessed dataset local storage
        model_with_emphasis_head: Model with emphasis detection capability
        hf_token: HuggingFace API token for accessing datasets
        ds_hf_train: HuggingFace dataset name for training (if applicable and the data is also used for training the emphasis detection head)
        ds_hf_eval: HuggingFace dataset name for evaluation (if applicable and the data is also used for evaluation of the emphasis detection head)
        emphasis_indices_column_name: Column name for emphasis labels
        columns_to_remove: Columns to exclude from the dataset
        split_train_val_percentage: Percentage of data to use for validation
    """
    
    def __init__(self, 
                preprocessed_dataset_path, 
                columns_to_remove, 
                model_with_emphasis_head, 
                hf_token=None,
                ds_hf_train=None, 
                ds_hf_eval=None,
                emphasis_indices_column_name="emphasis_indices", 
                transcription_column_name='transcription', 
                split_train_val_percentage=0.02,
                force_download=False,
                max_samples=None
            ):
        
        self.preprocessed_dataset_path = preprocessed_dataset_path
        self.model_with_emphasis_head = model_with_emphasis_head
        self.hf_token = hf_token
        self.ds_hf_train = ds_hf_train
        self.ds_hf_eval = ds_hf_eval
        self.emphasis_indices_column_name = emphasis_indices_column_name
        self.columns_to_remove = columns_to_remove
        self.transcription_column_name = transcription_column_name
        self.split_train_val_percentage = split_train_val_percentage
        self.force_download = force_download
        self.max_samples = max_samples

        # Clear cache before processing
        self.clear_dataset_cache(preprocessed_dataset_path)
        
        self.dataset = self.load_preproc_datasets(model_with_emphasis_head, 
                                                preprocessed_dataset_path, 
                                                columns_to_remove,
                                                emphasis_indices_column_name, 
                                                transcription_column_name, 
                                                ds_hf_train, 
                                                hf_token)

    def clear_dataset_cache(self, dataset_path):
        """Clear all cache files from dataset directory"""
        import glob
        import shutil
        
        if not os.path.exists(dataset_path):
            return
        
        # Find and remove all cache files
        cache_patterns = [
            os.path.join(dataset_path, "**", "cache-*.arrow"),
            os.path.join(dataset_path, "**", "tmp*"),
            os.path.join(dataset_path, "cache-*.arrow"),
            os.path.join(dataset_path, "tmp*")
        ]
        
        for pattern in cache_patterns:
            for cache_file in glob.glob(pattern, recursive=True):
                try:
                    if os.path.isfile(cache_file):
                        os.remove(cache_file)
                    elif os.path.isdir(cache_file):
                        shutil.rmtree(cache_file)
                    print(f"Removed cache: {cache_file}")
                except Exception as e:
                    print(f"Could not remove {cache_file}: {e}")

    def load_preproc_datasets(self, 
                    model_with_emphasis_head, 
                    preprocessed_dataset_path, 
                    columns_to_remove,
                    emphasis_indices_column_name, 
                    transcription_column_name,
                    ds_name_hf, 
                    hf_token):
        """
        Load and preprocess datasets from disk or HuggingFace.
        """
        def change_input_features(example):
            example['input_features'] = example['input_features'][0]
            return example
            
        def add_sentence_index(row, index_container):
            curr_index = index_container['sentence_index']
            row['sentence_index'] = curr_index
            index_container["sentence_index"] += 1
            return row
        
        # Skip local loading if force_download is True
        if os.path.exists(preprocessed_dataset_path) and not self.force_download:
            try:
                print(f"Loading preprocessed dataset from: {preprocessed_dataset_path}")
                
                # Clear any existing cache to avoid conflicts
                import shutil
                cache_dir = os.path.join(preprocessed_dataset_path, "train", "cache-*.arrow")
                import glob
                for cache_file in glob.glob(cache_dir):
                    try:
                        os.remove(cache_file)
                        print(f"Removed cache file: {cache_file}")
                    except:
                        pass
            
                train_set = load_from_disk(preprocessed_dataset_path)
                
                # LIMIT DATASET SIZE FOR DEBUGGING
                if self.max_samples is not None:
                    print(f"🔧 DEBUG MODE: Limiting dataset to first {self.max_samples} samples")
                    if isinstance(train_set, dict):
                        for split in train_set:
                            original_size = len(train_set[split])
                            train_set[split] = train_set[split].select(range(min(self.max_samples, original_size)))
                            print(f"  {split}: {original_size} → {len(train_set[split])} samples")
                    else:
                        original_size = len(train_set)
                        train_set = train_set.select(range(min(self.max_samples, original_size)))
                        print(f"  Dataset: {original_size} → {len(train_set)} samples")
                
                # Check dataset structure
                if isinstance(train_set, dict):
                    # Dataset with train/val/test splits
                    dataset_has_splits = 'train' in train_set
                    column_source = train_set['train'] if dataset_has_splits else train_set
                else:
                    # Flat dataset structure
                    dataset_has_splits = False
                    column_source = train_set
                
                # Check if wav2vec features need to be added
                if hasattr(model_with_emphasis_head, 'wav2vec_processor'):
                    
                    features_needed = False
                    temp_set = train_set.copy()  # Create a copy to avoid modifying the original
                    
                    # Handle different dataset structures
                    if dataset_has_splits:
                        for split in train_set:
                            if 'wav2vec_inputs' not in train_set[split].column_names:
                                print(f"❌ CRITICAL: wav2vec features missing in {split} split!")
                                temp_set[split] = add_wav2vec_features(train_set[split], model_with_emphasis_head)
                                features_needed = True
                            else:
                                print(f"✅ wav2vec features found in {split} split")
                    else:
                        if 'wav2vec_inputs' not in train_set.column_names:
                            print("❌ CRITICAL: wav2vec features missing in dataset!")
                            temp_set = add_wav2vec_features(train_set, model_with_emphasis_head)
                            features_needed = True
                        else:
                            print("✅ wav2vec features found in dataset")
                    
                    # Save updated dataset only if features were added
                    if features_needed:
                        temp_dir = preprocessed_dataset_path + "_temp"
                        temp_set.save_to_disk(temp_dir)
                        
                        # Delete original dataset and rename temp to original
                        import shutil
                        if os.path.exists(temp_dir):
                            shutil.rmtree(preprocessed_dataset_path)
                            os.rename(temp_dir, preprocessed_dataset_path)
                            train_set = temp_set  # Use the updated dataset
            
                # Remove unnecessary columns if specified
                if columns_to_remove:
                    # IMPORTANT: Preserve word_boundaries - don't remove them
                    safe_columns_to_remove = [col for col in columns_to_remove if col != "word_boundaries"]
                    
                    if isinstance(train_set, dict):
                        for split in train_set:
                            existing_cols = [col for col in safe_columns_to_remove if col in train_set[split].column_names]
                            if existing_cols:
                                train_set[split] = train_set[split].remove_columns(existing_cols)
                    else:
                        existing_cols = [col for col in safe_columns_to_remove if col in train_set.column_names]
                        if existing_cols:
                            train_set = train_set.remove_columns(existing_cols)
    
                return train_set
            except ValueError as e:
                # Check if error is about 'List' feature type
                if "Feature type 'List' not found" in str(e):
                    print(f"🔄 ERROR: Dataset contains 'List' feature type which is unsupported.")
                    print(f"🔄 Removing invalid dataset and recreating it...")
                    
                    # Remove the problematic dataset
                    import shutil
                    shutil.rmtree(preprocessed_dataset_path)
                    print(f"🔄 Deleted invalid dataset at {preprocessed_dataset_path}")
                    
                    # Force recreation of the dataset
                    print(f"🔄 Forcing recreation of dataset...")
                    self.force_download = True
                else:
                    # For other errors, re-raise
                    raise e

        # At this point, either force_download is True or the dataset doesn't exist or was deleted
        # Create new dataset directory if it doesn't exist
        os.makedirs(os.path.dirname(preprocessed_dataset_path), exist_ok=True)
        
        print(f"{'Forcing download from' if self.force_download else 'Creating new'} preprocessed dataset at {preprocessed_dataset_path}")
        ds_preprocessor = DSProcessor(
            ds_name=ds_name_hf,
            processor=model_with_emphasis_head.processor,
            hyperparameters={"split_train_val_percentage": self.split_train_val_percentage},
            hf_token=hf_token
        )
        print("DS Processor initialized.")
        train_set = ds_preprocessor.get_train_dataset(emphasis_indices_column_name=emphasis_indices_column_name, 
                                                       transcription_column_name=transcription_column_name,
                                                       model=model_with_emphasis_head,
                                                       columns_to_remove=[])
        
        # LIMIT DATASET SIZE FOR DEBUGGING - BEFORE PROCESSING
        if self.max_samples is not None:
            print(f"🔧 DEBUG MODE: Limiting dataset to first {self.max_samples} samples BEFORE processing")
            if isinstance(train_set, dict):
                for split in train_set:
                    original_size = len(train_set[split])
                    train_set[split] = train_set[split].select(range(min(self.max_samples, original_size)))
                    print(f"  {split}: {original_size} → {len(train_set[split])} samples")
            else:
                original_size = len(train_set)
                train_set = train_set.select(range(min(self.max_samples, original_size)))
                print(f"  Dataset: {original_size} → {len(train_set)} samples")
        
        # Add sentence indices
        index_container = {"sentence_index": 0}
        if isinstance(train_set, dict):
            # Process each split
            for split in train_set:
                train_set[split] = train_set[split].map(
                    add_sentence_index, 
                    num_proc=1, 
                    load_from_cache_file=False, 
                    fn_kwargs={'index_container': index_container}
                )
                train_set[split] = train_set[split].map(
                    change_input_features, 
                    load_from_cache_file=False, 
                    num_proc=1
                )
                # Rename labels if needed
                if "labels" in train_set[split].column_names:
                    train_set[split] = train_set[split].rename_column("labels", f"labels_{transcription_column_name}")
                
                # Add wav2vec features if the model supports it
                if hasattr(model_with_emphasis_head, 'wav2vec_processor'):
                    print(f"Adding wav2vec features to {split} split...")
                    train_set[split] = add_wav2vec_features(train_set[split], model_with_emphasis_head)
        else:
            # Process flat dataset
            train_set = train_set.map(
                add_sentence_index, 
                num_proc=1, 
                load_from_cache_file=False, 
                fn_kwargs={'index_container': index_container}
            )
            train_set = train_set.map(
                change_input_features, 
                load_from_cache_file=False, 
                num_proc=1
            )
            # Rename labels if needed
            if "labels" in train_set.column_names:
                train_set = train_set.rename_column("labels", f"labels_{transcription_column_name}")
            
            # Add wav2vec features if the model supports it
            if hasattr(model_with_emphasis_head, 'wav2vec_processor'):
                print("Adding wav2vec features to dataset...")
                train_set = add_wav2vec_features(train_set, model_with_emphasis_head)
        
        # After preprocessing and before saving:
        print("🔧 Converting list features to Sequence features for compatibility...")
        if isinstance(train_set, dict):
            for split in train_set:
                for col in train_set[split].column_names:
                    if train_set[split][0] is not None and col in train_set[split][0]:
                        if isinstance(train_set[split][0][col], list):
                            print(f"  Converting {col} from list to Sequence in {split} split")
                            train_set[split] = train_set[split].cast_column(
                                col, 
                                Sequence(feature=train_set[split].features[col].feature 
                                         if hasattr(train_set[split].features[col], 'feature') 
                                         else None)
                            )
        else:
            for col in train_set.column_names:
                if train_set[0] is not None and col in train_set[0]:
                    if isinstance(train_set[0][col], list):
                        print(f"  Converting {col} from list to Sequence")
                        train_set = train_set.cast_column(
                            col, 
                            Sequence(feature=train_set.features[col].feature 
                                     if hasattr(train_set.features[col], 'feature') 
                                     else None)
                        )

        # Save processed dataset
        print(f"💾 Saving processed dataset to {preprocessed_dataset_path}")
        train_set.save_to_disk(preprocessed_dataset_path)
        print(f"✅ Dataset saved successfully")
        
        # Remove unnecessary columns if specified
        if columns_to_remove:
            if isinstance(train_set, dict):
                for split in train_set:
                    existing_cols = [col for col in columns_to_remove if col in train_set[split].column_names]
                    if existing_cols:
                        train_set[split] = train_set[split].remove_columns(existing_cols)
            else:
                existing_cols = [col for col in columns_to_remove if col in train_set.column_names]
                if existing_cols:
                    train_set = train_set.remove_columns(existing_cols)
        
        return train_set

    def split_train_val(self):
        """
        Split dataset into training and validation sets.
        
        Args:
            rows_to_remove: Optional list of row indices to exclude
            
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        if self.split_train_val_percentage == 0.0:
            return self.dataset, None
        
        # Disable caching to avoid file system issues
        dataset_split = self.dataset["train"].train_test_split(
            test_size=self.split_train_val_percentage,
            shuffle=True,
            seed=42,
            load_from_cache_file=False,  # Disable caching
        )
        return dataset_split["train"], dataset_split["test"]
    
def load_data(model_with_emphasis_head, transcription_column_name, dataset_name, save_path=None, force_download=False, max_samples=None):
    """
    Factory function to create the appropriate dataset loader.
    
    *Add here any new datasets you want to support.*
    
    Args:
        model_with_emphasis_head: Model with emphasis detection capability
        transcription_column_name: Column name for transcription text
        dataset_name: Name of the dataset to load (e.g., "tinyStress-15K")
        save_path: Path to save or load the preprocessed dataset
        force_download: If True, forces downloading from HuggingFace even if local path exists
        
    Returns:
        Instantiated dataset loader for the specified dataset
        
    Raises:
        ValueError: If the requested dataset is not supported
    """
    dataset = None
    if dataset_name == "abinroy04/ITA-word-stress":
        dataset = PreprocessedITAStressLoader(
            model_with_emphasis_head, 
            transcription_column_name, 
            save_path=save_path,
            force_download=force_download,
            max_samples=max_samples
        )
    elif dataset_name == "abinroy04/ITA-timed":
        dataset = PreprocessedITATimeLoader(
            model_with_emphasis_head, 
            transcription_column_name, 
            save_path=save_path,
            force_download=force_download,
            max_samples=max_samples
        )
    else:
        raise ValueError(f"Dataset {dataset_name} is not defined in data_loader.py")

    return dataset

class PreprocessedITAStressLoader(PreprocessedDataLoader):
    """
    Data loader for the ITA-word-stress dataset.
    """
    def __init__(self, model_with_emphasis_head, transcription_column_name, save_path=None, force_download=False, max_samples=None):
        # Columns that are not needed for training or evaluation
        columns_to_remove = []
        
        # Path for saving preprocessed dataset
        if save_path is None:
            save_path = "./whistress_preprocessed_data"
        
        # Update the HF token with a valid one
        hf_token = os.environ.get("HF_TOKEN")
        
        # Initialize the parent class
        super().__init__(
            preprocessed_dataset_path=save_path,
            columns_to_remove=columns_to_remove,
            model_with_emphasis_head=model_with_emphasis_head,
            emphasis_indices_column_name="emphasis_indices",
            transcription_column_name=transcription_column_name,
            ds_hf_train="abinroy04/ITA-word-stress",
            hf_token=hf_token,
            force_download=force_download,
            max_samples=max_samples
        )

    def split_train_val_test(self):
        """
        Split ITA-word-stress dataset into train, validation, and test sets.
        """
        if self.split_train_val_percentage == 0.0:
            return self.dataset["train"], None, self.dataset["test"]
        train_set, eval_set = self.split_train_val()
        return train_set, eval_set, self.dataset["test"]

class PreprocessedITATimeLoader(PreprocessedDataLoader):
    """
    Data loader for the ITA-word-stress dataset.
    """
    def __init__(self, model_with_emphasis_head, transcription_column_name, save_path=None, force_download=False, max_samples=None):
        # Columns that are not needed for training or evaluation
        columns_to_remove = []
        
        # Path for saving preprocessed dataset
        if save_path is None:
            save_path = "./whistress_preprocessed_data"
        
        # Use the correct dataset name for ITA-timed
        hf_token = os.environ.get("HF_TOKEN")
        
        print("🔧 DEBUG: Initializing ITA-timed loader...")
        
        # Initialize the parent class
        super().__init__(
            preprocessed_dataset_path=save_path,
            columns_to_remove=columns_to_remove,
            model_with_emphasis_head=model_with_emphasis_head,
            emphasis_indices_column_name="emphasis_indices",
            transcription_column_name=transcription_column_name,
            ds_hf_train="abinroy04/ITA-timed",
            hf_token=hf_token,
            force_download=force_download,
            max_samples=max_samples
        )

    def split_train_val_test(self):
        """
        Split ITA-timed dataset into train, validation, and test sets.
        """
        # Debug the dataset structure before splitting
        print("🔧 DEBUG: Dataset structure before splitting:")
        print(f"  Dataset type: {type(self.dataset)}")
        print(f"  Dataset keys: {self.dataset.keys() if isinstance(self.dataset, dict) else 'Not a dict'}")
        
        if self.split_train_val_percentage == 0.0:
            return self.dataset["train"], None, self.dataset["test"]
        train_set, eval_set = self.split_train_val()
        return train_set, eval_set, self.dataset["test"]

def fix_complex_nested_lists(dataset):
    """
    Fix complex nested list structures that cause serialization issues.
    
    Args:
        dataset: Dataset with potential nested list/tensor issues
        
    Returns:
        Dataset with fixed serializable structures
    """
    def fix_map_dict(example):
        for col in example.keys():
            if isinstance(col, str) and col.startswith("map_dict_"):
                if isinstance(example[col], dict) and "values" in example[col]:
                    # Convert values to simple lists
                    values = example[col]["values"]
                    if isinstance(values, list):
                        fixed_values = []
                        for val in values:
                            if hasattr(val, 'tolist'):  # Handle torch.Tensor
                                fixed_values.append(val.tolist())
                            elif hasattr(val, '__iter__'):  # Handle other iterables
                                fixed_values.append(list(val))
                            else:  # Handle scalar values
                                fixed_values.append([val])
                        example[col]["values"] = fixed_values
        return example
    
    # Apply the fix to each example
    return dataset.map(
        fix_map_dict,
        desc="Fixing complex nested lists",
        num_proc=1
    )
