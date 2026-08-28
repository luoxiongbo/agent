from __future__ import annotations

from typing import Dict, Optional


class Registry:
    def __init__(self) -> None:
        self._skills: Dict[str, object] = {}
        self._scenes: Dict[str, object] = {}

    def register_skill(self, skill: object) -> None:
        self._skills[getattr(skill, "name")] = skill

    def register_scene(self, scene: object) -> None:
        self._scenes[getattr(scene, "name")] = scene

    def skill(self, name: str) -> Optional[object]:
        return self._skills.get(name)

    def scene(self, name: str) -> Optional[object]:
        return self._scenes.get(name)

    def skills(self) -> Dict[str, object]:
        return dict(self._skills)

    def scenes(self) -> Dict[str, object]:
        return dict(self._scenes)
