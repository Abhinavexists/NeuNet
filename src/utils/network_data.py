import json

def export_network(hidden_layer1, hidden_layer2, hidden_layer3, output_layer):
        network_data = {
            "layers": [
                {"type": "input", "neurons": 2},  # Input layer
                {"type": "hidden", "neurons": 32},  # Hidden layer 1
                {"type": "hidden", "neurons": 32},  # Hidden layer 2
                {"type": "hidden", "neurons": 16},  # Hidden layer 3
                {"type": "output", "neurons": 3},  # Output layer
            ],
            "connections": [],
        }

        for i, (weights, layer_name) in enumerate(
            [
                (hidden_layer1.weights, "hidden_layer1"),
                (hidden_layer2.weights, "hidden_layer2"),
                (hidden_layer3.weights, "hidden_layer3"),
                (output_layer.weights, "output_layer"),
            ]
        ):
            for source_neuron in range(weights.shape[0]):
                for target_neuron in range(weights.shape[1]):
                    connection = {
                        "source": {"layer": i, "neuron": source_neuron},
                        "target": {"layer": i + 1, "neuron": target_neuron},
                        "weight": weights[source_neuron, target_neuron],
                    }
                    network_data["connections"].append(connection)

        with open("src/utils/network_data.json", "w") as file:
            json.dump(network_data, file)
