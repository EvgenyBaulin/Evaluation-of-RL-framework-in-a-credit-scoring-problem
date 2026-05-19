"""
Training progress display using a single shared rich.Progress instance.

Both ExperimentProgress (outer, persists) and TrainingProgress (inner, transient)
add tasks to the same Progress so there is only one live display and no visual
fighting between two separate Progress contexts.

Usage:
    # Outer wrapper (optional — used by dimensionality runner):
    with ExperimentProgress(total=4) as exp:
        for dim in dims:
            with TrainingProgress(total_episodes=50, controller="dqn", dim=dim, seed=11) as prog:
                for ep in range(50):
                    prog.update(episode=ep, reward=reward)
            exp.update(label=f"dim={dim}")

    # Standalone (no outer wrapper — used by run_pipeline.py):
    with TrainingProgress(total_episodes=50, controller="dqn", dim=12, seed=11) as prog:
        for ep in range(50):
            prog.update(episode=ep, reward=reward)
"""
from __future__ import annotations

import time
from typing import Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console(stderr=False)

# ── Shared singleton ──────────────────────────────────────────────────────────
# One Progress instance for all tasks so there is only one live render loop.
# Columns use {task.fields[extra]} for flexible per-task text.

_shared: Optional[Progress] = None
_owners: int = 0          # reference count — started when first owner enters


def _get_progress() -> Progress:
    global _shared
    if _shared is None:
        _shared = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=38),
            MofNCompleteColumn(),
            TextColumn("[dim]{task.fields[extra]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=8,
            expand=False,
        )
    return _shared


def _acquire() -> None:
    global _owners
    p = _get_progress()
    if _owners == 0:
        p.start()
    _owners += 1


def _release() -> None:
    global _owners, _shared
    _owners -= 1
    if _owners == 0 and _shared is not None:
        _shared.stop()
        _shared = None


# ── ExperimentProgress ────────────────────────────────────────────────────────

class ExperimentProgress:
    """Persistent outer bar — tracks dimensions completing (e.g. 0/4 → 4/4)."""

    def __init__(self, total: int, title: str = "Dimensionality Experiment"):
        self.total = total
        self.title = title
        self._task: Optional[TaskID] = None

    def __enter__(self) -> "ExperimentProgress":
        _acquire()
        self._task = _get_progress().add_task(
            f"[bold magenta]{self.title}",
            total=self.total,
            extra="",
        )
        return self

    def update(self, label: str = "", advance: int = 1) -> None:
        if self._task is not None:
            _get_progress().update(self._task, advance=advance, extra=label)

    def __exit__(self, *args) -> None:
        if self._task is not None:
            _get_progress().update(self._task, extra="[green]done[/green]")
        _release()
        console.print(f"[bold green]✓ {self.title} complete.[/bold green]")


# ── TrainingProgress ──────────────────────────────────────────────────────────

class TrainingProgress:
    """Per-agent training bar — appears while training, disappears when done."""

    def __init__(
        self,
        total_episodes: int,
        controller: str,
        dim: int,
        scenario: str = "",
        seed: int = 0,
    ):
        self.total_episodes = total_episodes
        self.controller = controller
        self.dim = dim
        self.scenario = scenario
        self.seed = seed
        self._task: Optional[TaskID] = None
        self._start_time: float = 0.0
        self._last_reward: float = 0.0

    def __enter__(self) -> "TrainingProgress":
        _acquire()
        desc = f"[blue]{self.controller.upper()}[/blue] dim={self.dim}"
        if self.seed:
            desc += f" seed={self.seed}"
        self._task = _get_progress().add_task(
            desc,
            total=self.total_episodes,
            extra="",
        )
        self._start_time = time.time()
        return self

    def update(
        self,
        episode: int,
        reward: float = 0.0,
        loss: float = 0.0,
        advance: int = 1,
    ) -> None:
        self._last_reward = reward
        if self._task is not None:
            extra = f"[cyan]{reward:>10.1f}[/cyan] rew  [yellow]{loss:.4f}[/yellow] loss"
            _get_progress().update(self._task, advance=advance, extra=extra)

    def __exit__(self, *args) -> None:
        elapsed = time.time() - self._start_time
        if self._task is not None:
            _get_progress().remove_task(self._task)
            self._task = None
        # print() inside a live display renders above the bar without disrupting it
        console.print(
            f"[green]✓[/green] {self.controller.upper()} dim={self.dim} seed={self.seed} "
            f"— {elapsed:.1f}s  last reward=[cyan]{self._last_reward:.1f}[/cyan]"
        )
        _release()
