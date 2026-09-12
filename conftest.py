import sys
from pathlib import Path

import pytest

# Asegura que la raiz del proyecto este en sys.path
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    """Registra markers y filtros."""
    config.addinivalue_line(
        "markers",
        "network: test que hace peticiones HTTP a fuentes externas (se excluye por defecto)",
    )
    # Silencia DeprecationWarning de unicode_escape (contenido HTML externo).
    config.addinivalue_line(
        "filterwarnings",
        r"ignore:.*invalid escape sequence.*:DeprecationWarning",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="Ejecutar tambien los tests marcados @pytest.mark.network",
    )


def pytest_collection_modifyitems(config, items):
    """Excluye los tests 'network' salvo que se pase --run-network."""
    if config.getoption("--run-network"):
        return
    skip_network = pytest.mark.skip(reason="requiere --run-network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)
