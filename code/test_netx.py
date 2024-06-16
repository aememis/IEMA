import random
import networkx as nx
import ete3
from pyvis.network import Network
from matplotlib import pyplot as plt
from Individual import Individual

G = nx.DiGraph()

par1 = Individual("Parent 1")
par2 = Individual("Parent 2")
par3 = Individual("Parent 3")
par4 = Individual("Parent 4")
par5 = Individual("Parent 5")
par6 = Individual("Parent 6")

G.add_node(par1.id, data=par1.data)
G.add_node(par2.id, data=par2.data)
G.add_node(par3.id, data=par3.data)
G.add_node(par4.id, data=par4.data)
# G.add_node(par5.id, data=par5.data)
# G.add_node(par6.id, data=par6.data)

parents = [par1, par2, par3, par4]  # , par5, par6]
counter = 0

layers = {0: [par1.id, par2.id, par3.id, par4.id]}  # , par5.id, par6.id]}


def add_children(parents):
    global counter
    counter += 1
    random.shuffle(parents)
    children = []
    for i in range(0, len(parents), 2):
        new_child1 = Individual(random.randint(0, 10**3))
        G.add_node(new_child1.id, data=new_child1.data)
        new_child2 = Individual(random.randint(0, 10**3))
        G.add_node(new_child2.id, data=new_child2.data)
        children.append(new_child1)
        children.append(new_child2)

        G.add_edge(parents[i].id, new_child1.id)
        G.add_edge(parents[i + 1].id, new_child1.id)
        G.add_edge(parents[i].id, new_child2.id)
        G.add_edge(parents[i + 1].id, new_child2.id)

        if counter in layers:
            layers[counter].extend([new_child1.id, new_child2.id])
        else:
            layers[counter] = [new_child1.id, new_child2.id]

    if counter < 10:
        add_children(children)


add_children(parents)
# G.add_node(ind1.id, data=ind1.data)
# G.add_node(ind2.id, data=ind2.data)
# G.add_node(ind3.id)

# G.add_edge(ind1.id, ind3.id)
# G.add_edge(ind2.id, ind3.id)

# print the graph
print(G.nodes(data=True))
print(G.edges())

print(layers)
print(len(layers))
print(len(G.nodes()))

# define layers
# layers = {"a": [par1.id, par2.id, par3.id, par4.id]}
pos = nx.multipartite_layout(G, subset_key=layers)
nx.draw(
    G,
    pos,
    with_labels=False,
    node_color="lightblue",
    edge_color="gray",
    node_size=200,
    font_size=16,
)
plt.title("Graph with Multiple Parents")
plt.show()
