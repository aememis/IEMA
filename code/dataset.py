import json
import pickle

import config as cfg
import networkx as nx


class Dataset:
    pass


class FSD50K(Dataset):
    def __init__(self):
        with open(cfg.ONTOLOGY_GRAPH_PATH_FSD50K, "rb") as file:
            self.G = pickle.load(file)
        with open(cfg.ONTOLOGY_LAYERS_PATH_FSD50K, "r") as file:
            self.layers = json.load(file)
        self._create_super_node()
        self._create_bidirectional()

    def _create_super_node(self):
        super_node = -1
        self.G.add_node(super_node)
        for root in self.layers["0"]:
            self.G.add_edge(super_node, root)

    def _create_bidirectional(self):
        self.G_bi = self.G.copy()
        for u, v in self.G_bi.edges():
            self.G_bi.add_edge(v, u)

    def _is_under_same_parent(self, node1, node2, level):
        node1_parents = self.get_parents(node1, level=level)
        node2_parents = self.get_parents(node2, level=level)
        print(node1_parents)
        print(node2_parents)
        input()
        return any([p in node2_parents for p in node1_parents])

    def get_parents(self, node, level=None):
        level = str(level)
        parents = nx.ancestors(self.G, node)
        if level is not None:
            parents = [p for p in parents if p in self.layers[level]]
        input(("parents", parents))
        return set(parents)

    def get_nodes(self, level=None):
        if level is None:
            return self.G.nodes
        return self.layers[level]

    def get_distances(self, nodes1, nodes2):
        dists = []
        for node1 in nodes1:
            for node2 in nodes2:
                if self._is_under_same_parent(node1, node2, level="1"):
                    dists.append(0)
                    continue
                if nx.has_path(self.G, node1, node2):
                    dist = nx.shortest_path_length(self.G, node1, node2)
                    dists.append(dist)
                else:
                    raise ValueError(
                        f"No path between the nodes {node1} and {node2}"
                    )
                print(f"Distance between {node1} and {node2}: {dist}")
                input()
        return dists


if __name__ == "__main__":
    fsd50k = FSD50K()
    print(fsd50k.get_distances(["/t/dd00027"], ["/m/0kpv1t"]))
