import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_csv('intensity.csv')

report = classification_report(y_test, y_pred, zero_division=1)
lines = report.split('\n')

speakers = []
precisions = []
for line in lines[2:-3]:
    line = line.strip()
    if line:
        values = line.split()
        speaker = values[0]
        precision = float(values[1])
        speakers.append(speaker)
        precisions.append(precision)

G = nx.Graph()

for speaker in speakers:
    G.add_node(speaker)

for i in range(len(speakers)):
    for j in range(i + 1, len(speakers)):
        speaker1 = speakers[i]
        speaker2 = speakers[j]

        score1 = precisions[i]
        score2 = precisions[j]
        similarity = abs(score1 - score2)
        
        G.add_edge(speaker1, speaker2, weight=similarity)

plt.figure(figsize=(16, 12))
pos = nx.spring_layout(G)
labels = {speaker: speaker for speaker in speakers}
weights = nx.get_edge_attributes(G, 'weight')

nx.draw_networkx_nodes(G, pos, node_size=200, node_color='lightblue')
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
nx.draw_networkx_labels(G, pos, labels, font_size=10)
nx.draw_networkx_edge_labels(G, pos, edge_labels=weights, font_size=8)

plt.axis('off')
plt.show()
