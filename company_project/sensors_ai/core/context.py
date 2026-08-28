from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .result import Asset


@dataclass(frozen=True)
class AuditEntry:
    component: str
    action: str
    details: Dict[str, Any]


@dataclass
class AuditLog:
    entries: List[AuditEntry] = field(default_factory=list)

    def record(self, component: str, action: str, details: Dict[str, Any] | None = None) -> None:
        self.entries.append(AuditEntry(component=component, action=action, details=details or {}))


@dataclass(frozen=True)
class HistoryEntry:
    user_id: str
    goal: str
    output: str


@dataclass
class AssetStore:
    assets: List[Asset] = field(default_factory=list)

    def add(self, asset: Asset) -> Asset:
        self.assets.append(asset)
        return asset

    def list_by_type(self, asset_type: str) -> List[Asset]:
        return [asset for asset in self.assets if asset.asset_type == asset_type]


@dataclass
class ContextStore:
    knowledge: Dict[str, str] = field(default_factory=dict)
    memories: Dict[str, str] = field(default_factory=dict)
    assets: AssetStore = field(default_factory=AssetStore)
    audit_log: AuditLog = field(default_factory=AuditLog)
    history: List[HistoryEntry] = field(default_factory=list)

    def lookup_knowledge(self, query: str) -> Dict[str, str]:
        return {key: value for key, value in self.knowledge.items() if query in key or query in value}

    def lookup_memory(self, key: str) -> str | None:
        return self.memories.get(key)

    def record_history(self, user_id: str, goal: str, output: str) -> HistoryEntry:
        entry = HistoryEntry(user_id=user_id, goal=goal, output=output)
        self.history.append(entry)
        return entry

    def history_for(self, user_id: str) -> List[HistoryEntry]:
        return [entry for entry in self.history if entry.user_id == user_id]
