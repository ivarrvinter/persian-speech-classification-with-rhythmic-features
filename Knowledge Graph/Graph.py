from graphviz import Digraph

dot = Digraph()

dot.node("Load Data", shape="rectangle", style="filled", fillcolor="#FFCCCC")
dot.node("Split Data", shape="rectangle", style="filled", fillcolor="#CCCCFF")
dot.node("Feature Engineering", shape="rectangle", style="filled", fillcolor="#CCFFCC")
dot.node("Train Model", shape="rectangle", style="filled", fillcolor="#99FF99")
dot.node("Evaluate Model", shape="rectangle", style="filled", fillcolor="#CC99FF")
dot.node("X, y", shape="oval", style="filled", fillcolor="#FFFFCC")
dot.node("Classification Report", shape="oval", style="filled", fillcolor="#FFFFCC")

dot.edge("Load Data", "Split Data")
dot.edge("Split Data", "Feature Engineering")
dot.edge("Feature Engineering", "Train Model")
dot.edge("Train Model", "Evaluate Model")
dot.edge("Split Data", "X, y")
dot.edge("Evaluate Model", "Classification Report")

dot.render("knowledge_graph.gv", view=True)
