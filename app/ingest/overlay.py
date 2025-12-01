"""
Module overlay pour l'ingestion - Réexport du module unifié.

Ce module ne fait que réexporter la fonctionnalité du module
app.utils.image_annotation pour maintenir la compatibilité.
"""
from __future__ import annotations

# Réexport de la fonction unifiée
from app.utils.image_annotation import overlay_tag, TAGGED_DIR

__all__ = ["overlay_tag", "TAGGED_DIR"]