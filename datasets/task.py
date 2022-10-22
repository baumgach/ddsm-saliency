from typing import Any
from dataclasses import dataclass
from enum import Enum, auto


class TaskTarget(Enum):
    MULTICLASS_CLASSIFICATION = auto()
    BINARY_CLASSIFICATION = auto()
    MULTILABEL_CLASSIFICATION = auto()
    ORDINAL_REGRESSION = auto()
    BINARY_SEGMENTATION = auto()
    MULTILABEL_SEGMENTATION = auto()


@dataclass
class Task:
    support: Any
    query: Any
    task_target: TaskTarget
