"""Pipeline de treino do classificador de doenças em folhas de milho."""

DEFAULT_SEED = 42
"""Semente única para o split treino/validação e para o embaralhamento.

Fica no pacote, e não em um dos módulos, para que ``train.py`` possa usá-la como
default de argumento sem precisar importar o TensorFlow só por isso.
"""
