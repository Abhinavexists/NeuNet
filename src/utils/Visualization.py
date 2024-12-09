import json
import networkx as nx
import matplotlib.pyplot as plt

def visualize_network(json_file):
    # Load the network data from the JSON file
    with open(json_file, "r") as file:
        network_data = json.load(file)

    # Initialize the graph
    G = nx.DiGraph()

    # Add nodes for each layer
    layer_positions = {}
    y_offset = 0
    for layer_idx, layer in enumerate(network_data["layers"]):
        num_neurons = layer["neurons"]
        for neuron_idx in range(num_neurons):
            node_id = f"L{layer_idx}_N{neuron_idx}"
            G.add_node(node_id, layer=layer_idx)
        layer_positions[layer_idx] = y_offset
        y_offset += num_neurons

    # Add edges (connections) with weights as labels
    for connection in network_data["connections"]:
        source = f"L{connection['source']['layer']}_N{connection['source']['neuron']}"
        target = f"L{connection['target']['layer']}_N{connection['target']['neuron']}"
        weight = connection["weight"]
        G.add_edge(source, target, weight=round(weight, 2))

    # Define node positions for better visualization
    pos = nx.multipartite_layout(G, subset_key="layer")

    # Draw the graph
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="skyblue", alpha=0.8)
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=10, edge_color="gray")
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")

    # Add edge labels (weights)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    plt.title("Neural Network Visualization")
    plt.axis("off")
    plt.show()

# Run the visualization
if __name__ == "__main__":
    visualize_network("D:\\Neural-Network-from_Scratch\\visualisation\\network_data.json")
