# tools/render_erd.py
from sqlalchemy import create_engine, MetaData
from sqlalchemy_schemadisplay import create_schema_graph

# Pointe vers ton fichier SQLite (créé à partir de ton DDL)
engine = create_engine("sqlite:///./tmp.db")

metadata = MetaData()
metadata.reflect(bind=engine)

graph = create_schema_graph(
    metadata=metadata,
    show_datatypes=True,     # affiche les types de colonnes
    show_indexes=False,
    rankdir="LR",            # gauche -> droite
    concentrate=False
)

graph.write_png("images/schema.png")
print("Schéma généré => images/schema.png")
