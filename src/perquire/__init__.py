"""
Perquire: Reverse Embedding Search Through Systematic Questioning

A revolutionary AI system that reverses the traditional embedding search process.
Instead of finding embeddings that match a known query, Perquire investigates 
mysterious embeddings through systematic questioning, gradually uncovering what they represent.
"""

__version__ = "0.1.0"
__author__ = "Franklin Baldo"
__email__ = "franklinbaldo@gmail.com"

# Lightweight core - only expose provider factory functions
from .providers import list_available_providers

__all__ = [
    "list_available_providers",
]