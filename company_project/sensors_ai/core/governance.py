from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class GovernanceDecision:
    allowed: bool
    reason: str = ""


@dataclass
class Governance:
    permissions: Dict[str, List[str]] = field(default_factory=dict)

    def check(self, user_id: str, action: str) -> GovernanceDecision:
        if not action:
            return GovernanceDecision(allowed=True)
        if user_id not in self.permissions:
            return GovernanceDecision(allowed=True)
        if action in self.permissions[user_id]:
            return GovernanceDecision(allowed=True)
        return GovernanceDecision(allowed=False, reason=f"missing permission: {action}")
