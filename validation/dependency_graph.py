# -*- coding: utf-8 -*-
# validation/dependency_graph.py
# Fase 0: Analisis estatico de imports y dependencias entre modulos
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def get_imports(filepath):
    """Extrae los modulos locales importados en un archivo Python."""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    tree = ast.parse(content)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(('config.', 'indicators.', 'regimes.', 'src.', 'data.', 'validation.')):
                    imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith(('config', 'indicators', 'regimes', 'src', 'data', 'validation')):
                imports.append(node.module)
    return sorted(set(imports))

# Recorrer todos los .py relevantes
targets = []
for dirname in ['.', 'config', 'indicators', 'regimes', 'src', 'data', 'validation']:
    for root, dirs, files in os.walk(dirname):
        for f in files:
            if f.endswith('.py') and not f.startswith('_'):
                targets.append(os.path.join(root, f))

print('=== DEPENDENCIAS ENTRE MODULOS ===')
for path in sorted(targets):
    imports = get_imports(path)
    if imports:
        rel_path = os.path.relpath(path, ROOT)
        print(f'\n{rel_path}:')
        for imp in imports:
            print(f'  -> {imp}')
