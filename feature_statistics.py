#%%
from collections import defaultdict
from tqdm import tqdm
import torch
# from datasets import load_dataset
# from transformer_lens.utils import tokenize_and_concatenate
# from transformers import AutoTokenizer
# from sae_lens.load_model import load_model


class FeatureStatistics():
    def __init__(self, sae=None):
        self.highly_activating_indices = defaultdict(list)
        self.feature_to_clusters = defaultdict(list)
        self.cluster_to_features = defaultdict(list)
        self.feature_token_count = defaultdict(lambda: defaultdict(int))
        self.feature_word_count = defaultdict(lambda: defaultdict(int))
        self.top_boosted_logits = {}
        self.meta_feature_top_boosted_logits = {}
        self.n_features = sae.W_dec.shape[0] if sae is not None else 49152
        self.min_activation_values = torch.zeros(self.n_features, device='cpu')

    @torch.no_grad()
    def process_activating_examples(self, sae, model, dataset, batch_size=2):
        num_batches = len(dataset) // batch_size
        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size
            batch_tokens = dataset[start_idx:end_idx]

            
            batch = model.run_with_cache(
                batch_tokens,
                names_filter=[sae.cfg.hook_name],
                stop_at_layer=sae.cfg.hook_layer+1
            )[1][sae.cfg.hook_name]

            # remove the first token
            batch = batch[:, 1:]
            
            feature_acts = sae.encode(batch)

            # Create a mask for activations higher than the minimum
            mask = feature_acts > self.min_activation_values.unsqueeze(0).unsqueeze(0)
            
            # Use torch.where instead of torch.nonzero
            batch_nums, seq_positions, feature_indices = torch.where(mask)

            global_indices = start_idx + batch_nums.cpu().numpy()
            
            for i in range(len(feature_indices)):
                feature_idx = feature_indices[i].item()
                global_idx = global_indices[i]
                seq_pos = seq_positions[i].item()
                batch_num = batch_nums[i].item()
                
                activation_value = feature_acts[batch_num, seq_pos, feature_idx].item()
                
                if len(self.highly_activating_indices[feature_idx]) < 100:
                    self.highly_activating_indices[feature_idx].append((global_idx, seq_pos, activation_value))
                    if len(self.highly_activating_indices[feature_idx]) == 100:
                        self.highly_activating_indices[feature_idx].sort(key=lambda x: x[2], reverse=True)
                        self.min_activation_values[feature_idx] = self.highly_activating_indices[feature_idx][-1][2]
                else:
                    self.highly_activating_indices[feature_idx][-1] = (global_idx, seq_pos, activation_value)
                    self.highly_activating_indices[feature_idx].sort(key=lambda x: x[2], reverse=True)
                    self.min_activation_values[feature_idx] = self.highly_activating_indices[feature_idx][-1][2]

    @torch.no_grad()
    def process_clusters(self, sae, meta_sae, threshold=0.06):
        self.cluster_to_features = defaultdict(list)
        self.feature_to_clusters = defaultdict(list)
        for i in tqdm(range(sae.W_dec.shape[0])):
            x = sae.W_dec[i].unsqueeze(0)
            activations = meta_sae(x, threshold=threshold)["feature_acts"].squeeze()
            # Find the non-zero indices
            non_zero_indices = torch.nonzero(activations).squeeze(1)
            non_zero_acts = activations[non_zero_indices]
            self.feature_to_clusters[i] = [(idx.item(), act.item()) for idx, act in zip(non_zero_indices, non_zero_acts)]
            for idx, act in zip(non_zero_indices, non_zero_acts):
                self.cluster_to_features[idx.item()].append((i, act.item()))

    @torch.no_grad()
    def calculate_top_boosted_logits(self, sae, model, top_k=10):
        unembedding = model.W_U  # Assuming this is the unembedding matrix
        decoder_directions = sae.W_dec  # No need to transpose

        self.top_boosted_logits = {}
        
        for feature_idx in tqdm(range(decoder_directions.shape[0]), desc="Calculating top boosted logits"):
            # Ensure both tensors have the same dtype
            feature_direction = decoder_directions[feature_idx].to(unembedding.dtype)
            
            # Calculate logit contributions for this feature
            logit_contributions = feature_direction @ unembedding
            
            # Get the top k indices and values
            top_values, top_indices = torch.topk(logit_contributions, k=top_k)
            
            # Convert token indices to strings
            top_tokens = [model.to_string(idx.item()) for idx in top_indices]
            
            self.top_boosted_logits[feature_idx] = list(zip(top_tokens, top_values.tolist()))

    @torch.no_grad()
    def calculate_meta_top_boosted_logits(self, meta_sae, model, top_k=10):
        unembedding = model.W_U
        decoder_directions = meta_sae.W_dec

        self.meta_feature_top_boosted_logits = {}

        for feature_idx in tqdm(range(decoder_directions.shape[0]), desc="Calculating meta top boosted logits"):
            feature_direction = decoder_directions[feature_idx].to(unembedding.dtype)

            logit_contributions = feature_direction @ unembedding

            top_values, top_indices = torch.topk(logit_contributions, k=top_k)

            top_tokens = [model.to_string(idx.item()) for idx in top_indices]

            self.meta_feature_top_boosted_logits[feature_idx] = list(zip(top_tokens, top_values.tolist()))


    @torch.no_grad()
    def get_top_examples(self, feature_idx, tokenizer, dataset, top_k=10, pre_context=10, post_context=6):
        indices = self.highly_activating_indices[feature_idx]
        sorted_indices = indices[:top_k]  # Already sorted in get_highly_activating_examples
        
        top_examples = []
        for global_idx, seq_pos, activation in sorted_indices:
            seq_pos += 1
            context_start = max(0, seq_pos - pre_context)
            context_end = min(dataset.shape[1], seq_pos + post_context+1)
            context_tokens = dataset[global_idx, context_start:context_end]
            context_text = tokenizer.decode(context_tokens)
            
            example_info = {
                "token": dataset[global_idx, seq_pos].item(),
                "global_idx": global_idx,
                "seq_pos": seq_pos,
                "activation": activation,
                "context_text": context_text,
            }
            top_examples.append(example_info)
        
        return top_examples

    def calculate_token_and_word_counts(self, model, dataset):
        self.feature_token_count = defaultdict(lambda: defaultdict(int))
        self.feature_word_count = defaultdict(lambda: defaultdict(int))

        for feature_idx, activations in tqdm(self.highly_activating_indices.items()):
            for global_idx, seq_pos, _ in activations:
                token = model.to_string(dataset[global_idx, seq_pos+1].item()).lower().strip()
                self.feature_token_count[feature_idx][token] += 1
                
                word = self.get_full_word(model, dataset[global_idx], seq_pos+1).lower().strip()
                self.feature_word_count[feature_idx][word] += 1
            
            # Sort token counts for this feature
            self.feature_token_count[feature_idx] = dict(sorted(
                self.feature_token_count[feature_idx].items(),
                key=lambda item: item[1],
                reverse=True
            ))
            
            self.feature_word_count[feature_idx] = dict(sorted(
                self.feature_word_count[feature_idx].items(),
                key=lambda item: item[1],
                reverse=True
            ))

    def get_full_word(self, model, tokens, pos):
        word = model.to_string(tokens[pos].item())
        # Look backwards
        for i in range(pos - 1, -1, -1):
            prev_token = model.to_string(tokens[i].item())
            if prev_token.startswith(' '):  # New word marker in many tokenizers
                break
            word = prev_token + word
        # Look forwards
        for i in range(pos + 1, len(tokens)):
            next_token = model.to_string(tokens[i].item())
            if next_token.startswith(' '):
                break
            word += next_token
        return word.strip(' ')

    def save(self, filename):
        torch.save({
            'n_features': self.n_features,
            'highly_activating_indices': dict(self.highly_activating_indices),
            'feature_to_clusters': dict(self.feature_to_clusters) if hasattr(self, 'feature_to_clusters') else {},
            'cluster_to_features': dict(self.cluster_to_features) if hasattr(self, 'cluster_to_features') else {},
            'feature_token_count': dict(self.feature_token_count) if hasattr(self, 'feature_token_count') else {},
            'feature_word_count': dict(self.feature_word_count) if hasattr(self, 'feature_word_count') else {},
            'top_boosted_logits': self.top_boosted_logits if hasattr(self, 'top_boosted_logits') else {},
            'meta_feature_top_boosted_logits': self.meta_feature_top_boosted_logits if hasattr(self, 'meta_feature_top_boosted_logits') else {}
        }, filename)

    @classmethod
    def load(cls, filename, sae):
        data = torch.load(filename)
        stats = cls(sae)
        stats.n_features = data['n_features']
        stats.highly_activating_indices = defaultdict(list, data['highly_activating_indices'])
        stats.feature_to_clusters = defaultdict(list, data['feature_to_clusters'])
        stats.cluster_to_features = defaultdict(list, data['cluster_to_features'])
        stats.feature_token_count = defaultdict(lambda: defaultdict(int), data['feature_token_count'])
        stats.feature_word_count = defaultdict(lambda: defaultdict(int), data['feature_word_count'])
        stats.top_boosted_logits = data['top_boosted_logits']
        stats.meta_feature_top_boosted_logits = data['meta_feature_top_boosted_logits']
        return stats
    
