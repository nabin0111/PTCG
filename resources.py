import torch
import pandas as pd
import numpy as np
from collections import defaultdict

class PersonaResourceManager:
    def __init__(self, n_components=50, min_cluster_size=200):
        self.n_components = n_components
        self.min_cluster_size = min_cluster_size
        self.res = self._load_all()

    def _load_all(self):
        print("[-] Loading Cluster Resources...")
        
        p_dict = torch.load("./data/persona_hub/persona_embeddings.pt", weights_only=False)
        t_dict = torch.load("./data/persona_hub/test_text_embeddings.pt", weights_only=False)
        df = pd.read_csv(f"./data/persona_hub/persona_clusters_{self.n_components}d_{self.min_cluster_size}_no_-1.csv")

        original_persona_list = p_dict['personas']
        original_persona_embeddings = p_dict['embeddings']
        cluster_personas = set(df['persona'])

        selected_indices = [
            idx for idx, persona in enumerate(original_persona_list)
            if persona in cluster_personas
        ]
        p_list = [original_persona_list[i] for i in selected_indices]
        p_emb = original_persona_embeddings[selected_indices]

        label_dict = {row['persona']: row['cluster_label'] for _, row in df.iterrows()}
        label_to_p = defaultdict(list)
        for p, l in label_dict.items(): 
            label_to_p[l].append(p)

        c_center = pd.read_csv(f"./data/persona_hub/cluster_centroid_representatives_{self.n_components}d_{self.min_cluster_size}.csv").set_index('cluster_label')['centroid_persona'].to_dict()
        c_dist = pd.read_csv(f"./data/persona_hub/cluster_centroid_cosine_distances_{self.n_components}d_{self.min_cluster_size}.csv").values
        
        c_dict = torch.load(f"./data/persona_hub/cluster_centroids_{self.n_components}d_{self.min_cluster_size}.pt", weights_only=False)
        centroid_embeddings = c_dict["embeddings"]

        c_vectors = torch.stack(centroid_embeddings) if isinstance(centroid_embeddings, list) else centroid_embeddings

        return {
            "p_list": p_list, 
            "p_emb": p_emb,
            "t_emb": t_dict['embeddings'], 
            "label_dict": label_dict,
            "label_to_p": label_to_p, 
            "c_center": c_center, 
            "c_dist": c_dist, 
            "c_vectors": c_vectors,
            "label_list": sorted(list(label_to_p.keys()))
        }

    def select_personas(self, method, n, auth_c_idx=None):
        if method == "cluster_diff_case" and auth_c_idx is not None:
            dist_row = self.res["c_dist"][auth_c_idx]
            sorted_idx = np.argsort(dist_row).tolist()

            num_similar = (n - 1) // 2
            num_dissimilar = n - 1 - num_similar
            
            sel_idx = [sorted_idx[0]] + sorted_idx[1:num_similar + 1] + sorted_idx[-num_dissimilar:]
            
            return [self.res["c_center"][self.res["label_list"][i]] for i in sel_idx[:n]]
        
        elif method == "cluster_random":
            sel_labels = np.random.choice(self.res["label_list"], n, replace=False)
            return [self.res["c_center"][l] for l in sel_labels]
        
        elif method == "wo_persona":
            return [None for _ in range(n)]
        
        return np.random.choice(self.res["p_list"], n, replace=False).tolist()