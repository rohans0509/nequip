# src/managers/logging_manager.py
"""
LoggingManager module:
This module sets up logging with enhanced Rich formatting and provides helper functions
for logging messages, displaying progress bars, and printing formatted tables.
"""

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn
from rich.theme import Theme
from rich.table import Table
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

class LoggingManager:
    def __init__(self, log_file: Optional[str] = None):
        # Create a Rich console with a custom theme.
        self.theme = Theme({
            "info": "bold cyan",
            "warning": "bold yellow",
            "error": "bold red",
            "success": "bold green",
            "highlight": "bold magenta",
            "muted": "dim white",
            "table.header": "bold blue",
            "progress.description": "bold cyan",
            "progress.percentage": "bold green",
            "progress.remaining": "bold yellow"
        })
        
        self.console = Console(theme=self.theme)
        # Configure logging handlers.
        handlers = [
            RichHandler(
                console=self.console,
                rich_tracebacks=True,
                show_time=True,
                show_path=False
            )
        ]
        if log_file:
            handlers.append(logging.FileHandler(log_file))
        else:
            handlers.append(logging.NullHandler())
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=handlers
        )
        self.logger = logging.getLogger("rich")

    def info(self, message: str) -> None:
        """Log an informational message."""
        self.console.print(message)

    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.console.print(f"⚠️  {message}", style="warning")

    def error(self, message: str) -> None:
        """Log an error message."""
        self.console.print(f"❌ {message}", style="error")

    def success(self, message: str) -> None:
        """Log a success message."""
        self.console.print(f"✅ {message}", style="success")

    def create_progress(self) -> Progress:
        """Create a Rich progress bar."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="green", finished_style="bold green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        )

    def create_table(self, title: str, columns: List[str]) -> Table:
        """Create a table for logging data."""
        table = Table(title=title, show_header=True, header_style="table.header")
        for column in columns:
            table.add_column(column, justify="center")
        return table

    def log_dict(self, data: Dict[str, Any], title: str = "Configuration") -> None:
        """Log a dictionary as a formatted table."""
        table = self.create_table(title, ["Parameter", "Value"])
        for key, value in data.items():
            table.add_row(str(key), str(value))
        self.console.print(table)

    def log_metrics(self, metrics: Dict[str, float], title: str = "Metrics") -> None:
        """Log metrics as a formatted table."""
        table = self.create_table(title, ["Metric", "Value"])
        for metric, value in metrics.items():
            formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            table.add_row(metric.replace("_", " ").title(), formatted_value)
        self.console.print(table)

    def section(self, title: str) -> None:
        """Print a section header."""
        self.console.print(f"\n[highlight]{'='*20} {title} {'='*20}[/]\n")

    def divider(self) -> None:
        """Print a divider line."""
        self.console.print("[muted]" + "-" * 80 + "[/]")
