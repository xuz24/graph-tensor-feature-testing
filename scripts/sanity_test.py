import torch

graph = torch.load("data/graphs/dp_drugs/unified_drugcomb_hetero.pt")
print("Graph type:", type(graph))
print("\nGraph attributes:")
for key in dir(graph):
    if not key.startswith('_'):
        print(f"  {key}")

print("\nEdge types:")
if hasattr(graph, 'edge_types'):
    for edge_type in graph.edge_types:
        print(f"  {edge_type}: {graph[edge_type]}")

print("\nNode types:")
if hasattr(graph, 'node_types'):
    for node_type in graph.node_types:
        print(f"  {node_type}")

print("\nDirect keys:")
print(graph.keys())