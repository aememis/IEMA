import json
import pickle

import config as cfg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


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
        # self.draw_graph(self.G_bi)

    def _create_super_node(self):
        super_node = -1
        self.G.add_node(super_node)
        self.G.nodes[super_node]["label"] = "super node"
        for root in self.layers["1"]:
            self.G.add_edge(super_node, root)
        self.layers["0"] = [super_node]

    def _create_bidirectional(self):
        self.G_bi = self.G.copy()
        for u, v in self.G_bi.edges():
            self.G_bi.add_edge(v, u)

    def _is_under_same_parent(self, node1, node2, level):
        node1_parents = self.get_parents(node1, level=level)
        node2_parents = self.get_parents(node2, level=level)
        return any([p in node2_parents for p in node1_parents])

    def get_parents(self, node, level=None):
        if node not in self.G.nodes:
            raise ValueError(f"Node {node} not in the graph")
        parents = nx.ancestors(self.G, node)
        if level is not None:
            level = str(level)
            parents = [p for p in parents if p in self.layers[level]]
        return set(parents)

    def get_nodes(self, level=None):
        if level is None:
            return self.G.nodes
        return self.layers[level]

    def is_parent_of_each_other(self, node1, node2):
        node1_parents = self.get_parents(node1)
        if node2 in node1_parents:
            return True
        node2_parents = self.get_parents(node2)
        if node1 in node2_parents:
            return True
        return False

    def get_level(self, node):
        for level, nodes in self.layers.items():
            if node in nodes:
                return level
        return None

    def get_subgraph(self, max_level):
        """get the subgraph of the ontology in or below the given level"""
        subgraph_levels = [str(i) for i in range(int(max_level) + 1)]
        subgraph_nodes = set()
        for l in subgraph_levels:
            subgraph_nodes.update(self.layers[l])
        return self.G_bi.subgraph(subgraph_nodes).copy()

    def get_distances(self, nodes1, nodes2):
        max_level = max(
            [int(self.get_level(n)) for n in np.concatenate([nodes1, nodes2])]
        )
        G_bi_sub = self.get_subgraph(max_level)
        dists = []
        for node1 in nodes1:
            for node2 in nodes2:
                if self._is_under_same_parent(node1, node2, level="1"):
                    dists.append(0)
                    continue
                if nx.has_path(G_bi_sub, node1, node2):
                    dist = nx.shortest_path_length(G_bi_sub, node1, node2)
                    dists.append(dist)
                else:
                    raise ValueError(
                        f"No path between the nodes {node1} and {node2}"
                    )
        return dists

    def draw_graph(self, G):
        sub_layers = {}
        for k, vs in self.layers.items():
            for v in vs:
                if v in G.nodes:
                    if k not in sub_layers:
                        sub_layers[k] = []
                    sub_layers[k].append(v)

        plt.clf()
        pos = nx.multipartite_layout(
            G, subset_key=sub_layers, align="horizontal"
        )
        nx.draw(
            G,
            pos,
            # labels=nx.get_node_attributes(G, "label"),
            # with_labels=True,
            node_size=75,
            font_size=10,
            font_weight="bold",
            arrowsize=5,
            width=1,
            alpha=0.75,
        )

        # Add labels
        for node in G.nodes():
            x, y = pos[node]
            label = G.nodes[node]["label"]
            plt.text(
                x,
                y,
                label[:20],
                fontsize=8,
                rotation=45,
                ha="left",
            )

        plt.show()


if __name__ == "__main__":
    fsd50k = FSD50K()
    print(fsd50k.get_distances(["/t/dd00027"], ["/m/0kpv1t"]))
