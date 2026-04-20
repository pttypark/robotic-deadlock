from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


Cell = Tuple[int, int]


class SystemType(str, Enum):
    DSD = "dsd"
    FSD = "fsd"


class TransactionStatus(str, Enum):
    WAITING = "waiting"
    ASSIGNED = "assigned"
    SERVICING = "servicing"
    COMPLETED = "completed"


class TransactionKind(str, Enum):
    STORAGE = "storage"
    RETRIEVAL = "retrieval"


class TransactionPhase(str, Enum):
    TO_BUFFER = "to_buffer"
    TO_STORAGE = "to_storage"


@dataclass
class ConflictZone:
    id: str
    center: Cell
    cells: Set[Cell]
    decision_points: Set[Cell]
    waiting_points: Set[Cell]
    escape_points: Set[Cell]


@dataclass
class PointMap:
    layout: str
    aisle_columns: List[int]
    transition_rows: List[int]
    storage_points: List[Cell]
    buffer_points: List[Cell]
    decision_points: Set[Cell]
    waiting_points: Set[Cell]
    escape_points: Set[Cell]
    conflict_zones: List[ConflictZone]
    zones: List[Set[Cell]]

    @property
    def width(self) -> int:
        return len(self.layout.strip().replace(" ", "").splitlines()[0])

    @property
    def height(self) -> int:
        return len(self.layout.strip().replace(" ", "").splitlines())

    @property
    def highway_points(self) -> Set[Cell]:
        points = _layout_highway_points(self.layout)
        points.update(self.storage_points)
        points.update(self.buffer_points)
        points.update(self.decision_points)
        points.update(self.waiting_points)
        return points


def _layout_highway_points(layout: str) -> Set[Cell]:
    points = set()
    lines = layout.strip().replace(" ", "").splitlines()
    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            if char.lower() in {"g", "."}:
                points.add((x, y))
    return points


@dataclass
class Transaction:
    id: int
    target: Cell
    zone_id: int
    created_step: int
    kind: TransactionKind = TransactionKind.RETRIEVAL
    buffer: Optional[Cell] = None
    phase: TransactionPhase = TransactionPhase.TO_STORAGE
    assigned_step: Optional[int] = None
    service_started_step: Optional[int] = None
    completed_step: Optional[int] = None
    assigned_agent: Optional[int] = None
    status: TransactionStatus = TransactionStatus.WAITING

    def __post_init__(self):
        if self.buffer is None:
            self.buffer = self.target
        if self.kind == TransactionKind.STORAGE:
            self.phase = TransactionPhase.TO_BUFFER
        else:
            self.phase = TransactionPhase.TO_STORAGE

    def current_target(self) -> Cell:
        if self.phase == TransactionPhase.TO_BUFFER:
            return self.buffer
        return self.target

    def waiting_time(self, step: int) -> int:
        end = self.assigned_step if self.assigned_step is not None else step
        return max(0, end - self.created_step)

    def flow_time(self) -> Optional[int]:
        if self.completed_step is None:
            return None
        return self.completed_step - self.created_step


@dataclass
class TransactionLedger:
    transactions: Dict[int, Transaction] = field(default_factory=dict)
    next_id: int = 1

    def create(
        self,
        target: Cell,
        zone_id: int,
        step: int,
        kind: TransactionKind = TransactionKind.RETRIEVAL,
        buffer: Optional[Cell] = None,
    ) -> Transaction:
        tx = Transaction(
            id=self.next_id,
            target=target,
            zone_id=zone_id,
            created_step=step,
            kind=kind,
            buffer=buffer,
        )
        self.transactions[tx.id] = tx
        self.next_id += 1
        return tx

    def waiting(self) -> List[Transaction]:
        return [
            tx
            for tx in self.transactions.values()
            if tx.status == TransactionStatus.WAITING
        ]

    def active(self) -> List[Transaction]:
        return [
            tx
            for tx in self.transactions.values()
            if tx.status in {TransactionStatus.ASSIGNED, TransactionStatus.SERVICING}
        ]

    def completed(self) -> List[Transaction]:
        return [
            tx
            for tx in self.transactions.values()
            if tx.status == TransactionStatus.COMPLETED
        ]

    def assign(self, tx: Transaction, agent_idx: int, step: int) -> None:
        tx.assigned_agent = agent_idx
        tx.assigned_step = step
        tx.status = TransactionStatus.ASSIGNED

    def start_service(self, tx: Transaction, step: int) -> None:
        tx.service_started_step = step
        tx.status = TransactionStatus.SERVICING

    def advance_after_service(self, tx: Transaction, step: int) -> bool:
        if tx.kind == TransactionKind.STORAGE and tx.phase == TransactionPhase.TO_BUFFER:
            tx.phase = TransactionPhase.TO_STORAGE
            tx.service_started_step = None
            tx.status = TransactionStatus.ASSIGNED
            return False
        if tx.kind == TransactionKind.RETRIEVAL and tx.phase == TransactionPhase.TO_STORAGE:
            tx.phase = TransactionPhase.TO_BUFFER
            tx.service_started_step = None
            tx.status = TransactionStatus.ASSIGNED
            return False
        self.complete(tx, step)
        return True

    def complete(self, tx: Transaction, step: int) -> None:
        tx.completed_step = step
        tx.status = TransactionStatus.COMPLETED
