import random
import time

import pandas as pd

from numpy import NaN

from rich.spinner import Spinner
from rich.console import Console
from rich.live import Live
from rich.table import Table

console = Console()

# Function to update the table
def update_table(data):
    table = Table(width=64)
    table.add_column("Iteration", justify="right", style="cyan", no_wrap=True)
    table.add_column(f"Progress\n(% of target)", justify="center", style="magenta")
    table.add_column("Queries\n(accepted/total)", justify="center", style="magenta")
    table.add_column("Threshold", justify="center", style="green")
    if len(data) > 5:
        table.add_row("...", "...", "...", "...")
    for i in range(len(data))[-5:]:
        table.add_row(
            str(data["iteration"].iloc[i]),
            f"{data['progress'].iloc[i]}%",
            f"{data['accepted_queries'].iloc[i]}/{data['total_queries'].iloc[i]}",
            f"{data['threshold'].iloc[i]}",
        )
    console.clear()
    console.print(table)

target = 50
items_in_beginning = 100

data = pd.DataFrame(columns=["iteration", "progress", "total_queries", "accepted_queries", "threshold"])
with console.status("Processing...", spinner="dots") as status:
    with Live(update_table(data), refresh_per_second=4) as live:
        for i in range(10):
            threshold = data['threshold'].iloc[-1] if i > 0 else 0.81
            data.loc[i] = [i, "--.-", items_in_beginning - sum(data["accepted_queries"]), "---", "-.--"]
            
            live.update(update_table(data))
            time.sleep(2)
            accepted = random.randint(1, 10)
            data.loc[i, "accepted_queries"] = accepted
            data.loc[i, "progress"] = round(100*sum(data["accepted_queries"]) / target, 1)
            data.loc[i, "threshold"] = round(threshold - 0.1 if accepted < 1 else threshold,2)
            live.update(update_table(data))
            time.sleep(2)