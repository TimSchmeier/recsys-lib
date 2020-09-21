from annoy import AnnoyIndex
import numpy as np


def build_index(vectors, normalize=True, num_trees=20):
    dim = vectors.shape[1]
    t = AnnoyIndex(dim, "angular")
    for vid, v in enumerate(vectors.tolist()):
        if normalize:
            v = np.asarray(v) / np.linalg.norm(v)
        t.add_item(vid, v)
    t.build(num_trees)
    return t


class NNS:
    def __init__(self):
        self.annoy_indices = {}
        self.id_to_index_maps = {}
        self.index_to_id_maps = {}
        self.entities = []

    def add_index(self, annoy_index, id_to_index_map, entity_name):
        self.annoy_indices[entity_name] = annoy_index
        self.id_to_index_maps[entity_name] = id_to_index_map
        index_to_id_map = {v: k for k, v in id_to_index_map.items()}
        self.index_to_id_maps[entity_name] = index_to_id_map
        self.entities.append(entity_name)

    def get_entity_index(self, entity_name, entity_id):
        return self.id_to_index_maps[entity_name][entity_id]

    def get_entity_id(self, entity_name, entity_index):
        return self.index_to_id_maps[entity_name][entity_index]

    def get_entity_vector(self, entity_name, entity_id):
        entity_index = self.get_entity_index(entity_name, entity_id)
        return self.annoy_indices[entity_name].get_item_vector(entity_index)

    def query_by_vector(self, entity_name, vector, num_nns):
        nns = self.annoy_indices[entity_name].get_nns_by_vector(
            vector, num_nns
        )
        return [self.index_to_id_maps[entity_name][n] for n in nns]

    def get_similar(self, entity_name, entity_id, num_nns, match_entity):
        v = self.get_entity_vector(entity_name, entity_id)
        return self.query_by_vector(match_entity, v, num_nns)
