from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn
from rich.theme import Theme
from rich.table import Table
from rich.style import Style
from rich.text import Text
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

class LoggingManager:
    def __init__(self, log_file: Optional[str] = None):
        # Create rich console with enhanced custom theme
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

        # Configure logging with enhanced formatting
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[
                RichHandler(
                    console=self.console,
                    rich_tracebacks=True,
                    show_time=True,
                    show_path=False
                ),
                logging.FileHandler(log_file) if log_file else logging.NullHandler()
            ]
        )
        self.logger = logging.getLogger("rich")

    def info(self, message: str) -> None:
        self.logger.info(Text(message, style="info"))

    def warning(self, message: str) -> None:
        self.logger.warning(Text(message, style="warning"))

    def error(self, message: str) -> None:
        self.logger.error(Text(message, style="error"))

    def success(self, message: str) -> None:
        self.logger.info(Text(message, style="success"))

    def create_progress(self) -> Progress:
        """Create an enhanced progress bar."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="green", finished_style="bold green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        )

    def create_table(self, title: str, columns: List[str]) -> Table:
        """Create a rich formatted table."""
        table = Table(title=title, show_header=True, header_style="table.header")
        for column in columns:
            table.add_column(column, justify="center")
        return table

    def log_dict(self, data: Dict[str, Any], title: str = "Configuration") -> None:
        """Log dictionary as a formatted table."""
        table = self.create_table(title, ["Parameter", "Value"])
        for key, value in data.items():
            table.add_row(str(key), str(value))
        self.console.print(table)

    def log_metrics(self, metrics: Dict[str, float], title: str = "Metrics") -> None:
        """Log metrics as a formatted table."""
        table = self.create_table(title, ["Metric", "Value"])
        for metric, value in metrics.items():
            formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            table.add_row(
                metric.replace("_", " ").title(),
                formatted_value,
                style="success" if "accuracy" in metric.lower() else None
            )
        self.console.print(table)

    def section(self, title: str) -> None:
        """Print a section header."""
        self.console.print(f"\n[highlight]{'='*20} {title} {'='*20}[/]\n")

    def divider(self) -> None:
        """Print a divider line."""
        self.console.print("[muted]" + "-" * 80 + "[/]") 