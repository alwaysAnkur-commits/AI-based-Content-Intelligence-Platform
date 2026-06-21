# src/knowledge_graph.py
import networkx as nx
from pyvis.network import Network
import pandas as pd

ENTITY_COLOR_MAP = {"PERSON": "#FF6B6B", "ORG": "#4ECDC4", "GPE": "#FFD93D",
                     "DATE": "#A8DADC", "MONEY": "#95E06C", "default": "#CCCCCC"}

def build_knowledge_graph(triplets_df: pd.DataFrame, entities_df: pd.DataFrame, min_freq: int = 2) -> nx.DiGraph:
    G = nx.DiGraph()
    entity_label_map = entities_df.groupby("entity_text")["entity_label"].first().to_dict()

    edge_counts = triplets_df.groupby(["subject", "object", "mapped_relation_type"]).size().reset_index(name="count")
    edge_counts = edge_counts[edge_counts["count"] >= min_freq]

    for _, row in edge_counts.iterrows():
        G.add_node(row["subject"], label=entity_label_map.get(row["subject"], "default"))
        G.add_node(row["object"], label=entity_label_map.get(row["object"], "default"))
        G.add_edge(row["subject"], row["object"], relation=row["mapped_relation_type"], weight=row["count"])

    print(f"Knowledge graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def visualize_knowledge_graph(G: nx.DiGraph, output_path: str = "data/processed/knowledge_graph.html"):
    net = Network(height="800px", width="100%", directed=True, notebook=False)
    for node, attrs in G.nodes(data=True):
        color = ENTITY_COLOR_MAP.get(attrs.get("label", "default"), ENTITY_COLOR_MAP["default"])
        net.add_node(node, label=node, color=color, title=attrs.get("label", ""))
    for source, target, attrs in G.edges(data=True):
        net.add_edge(source, target, label=attrs.get("relation", ""), value=attrs.get("weight", 1))
    net.show_buttons(filter_=["physics"])
    net.write_html(output_path)
    print(f"Saved interactive graph to {output_path}")

if __name__ == "__main__":
    triplets_df = pd.read_csv("data/processed/relation_triplets.csv")
    entities_df = pd.read_csv("data/processed/entities.csv")
    G = build_knowledge_graph(triplets_df, entities_df)
    visualize_knowledge_graph(G)