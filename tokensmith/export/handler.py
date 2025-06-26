from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ..manager import DatasetManager  # only used for type hints, won't cause import loop

class ExportHandler:
    def __init__(self, manager: 'DatasetManager'):
        self.manager = manager