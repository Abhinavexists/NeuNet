import json
import networkx as nx
import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd
import numpy as np

def network_visualization(json_file):
    # Load the network data from the JSON file
    with open(json_file, "r") as file:
        network_data = json.load(file)
    
    # Create a NetworkX graph
    G = nx.DiGraph()
    
    # Collect nodes and edges
    nodes = []
    edges = []
    
    # Track layer information
    layer_info = network_data["layers"]
    
    # Create node positions
    for layer_idx, layer in enumerate(layer_info):
        num_neurons = layer["neurons"]
        
        # Distribute neurons vertically in the layer
        y_positions = np.linspace(0, 1, num_neurons)
        
        for neuron_idx in range(num_neurons):
            node_id = f"L{layer_idx}_N{neuron_idx}"
            
            # Clamp color values between 0 and 255
            red = min(255, max(0, 50 * (layer_idx + 1)))
            green = min(255, max(0, (100 * (layer_idx + 1)) % 255))
            blue = min(255, max(0, 200 - (50 * (layer_idx + 1)) % 255))
            opacity = 0.7  # Fixed opacity value
            
            # Create node with position
            nodes.append({
                'id': node_id,
                'layer': layer_idx,
                'neuron_idx': neuron_idx,
                'x': layer_idx,
                'y': y_positions[neuron_idx],
                'size': 10,
                'color': f'rgba({red}, {green}, {blue}, {opacity})'
            })
    
    # Process connections
    connection_details = []
    for connection in network_data["connections"]:
        source = f"L{connection['source']['layer']}_N{connection['source']['neuron']}"
        target = f"L{connection['target']['layer']}_N{connection['target']['neuron']}"
        weight = connection["weight"]
        
        # Create edge
        edges.append({
            'source': source,
            'target': target,
            'weight': weight
        })
        
        # Collect connection details for hover information
        connection_details.append({
            'Source Layer': connection['source']['layer'],
            'Source Neuron': connection['source']['neuron'],
            'Target Layer': connection['target']['layer'],
            'Target Neuron': connection['target']['neuron'],
            'Weight': round(weight, 4)
        })
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add edges (connections)
    for edge in edges:
        source_node = next(node for node in nodes if node['id'] == edge['source'])
        target_node = next(node for node in nodes if node['id'] == edge['target'])
        
        # Color edges based on weight
        edge_color = 'green' if edge['weight'] > 0 else 'red'
        
        # Add edge with varying width based on weight magnitude
        fig.add_trace(go.Scatter(
            x=[source_node['x'], target_node['x']],
            y=[source_node['y'], target_node['y']],
            mode='lines',
            line=dict(
                color=edge_color,
                width=abs(edge['weight']) * 5  # Adjust line width based on weight
            ),
            opacity=0.5,  # Set opacity here at trace level
            hoverinfo='text',
            hovertext=f"Weight: {edge['weight']:.4f}",
            showlegend=False
        ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=[node['x'] for node in nodes],
        y=[node['y'] for node in nodes],
        mode='markers+text',
        marker=dict(
            size=[node['size'] for node in nodes],
            color=[node['color'] for node in nodes],
            line=dict(width=1, color='DarkSlateGrey')
        ),
        text=[node['id'] for node in nodes],
        textposition='top center',
        hoverinfo='text',
        hovertext=[f"Node: {node['id']}<br>Layer: {node['layer']}<br>Neuron Index: {node['neuron_idx']}" for node in nodes]
    ))
    
    # Update layout for better visualization
    fig.update_layout(
        title='Neural Network Visualization',
        title_x=0.5,
        width=1200,
        height=800,
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='white',
        xaxis=dict(
            title='Layer Depth',
            showgrid=True,
            zeroline=False,
            showticklabels=True
        ),
        yaxis=dict(
            title='Neuron Position',
            showgrid=True,
            zeroline=False,
            showticklabels=True
        )
    )
    
    # Save as interactive HTML
    pio.write_html(fig, file='neural_network_visualization.html', auto_open=True)
    
    return fig

# Usage
if __name__ == "__main__":
    try:
        interactive_fig = network_visualization(
            "D:\\Neural-Network-from_Scratch\\src\\utils\\network_data.json"
        )
    except Exception as e:
        print(f"Visualization error: {e}")
