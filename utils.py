def log_print(log_name: str, content: str, num_bar: int = 50) -> None:
    """Print a progress bar for a specific step.

    Args:
        log_name (str): The name of the log.
        content (str): The content to display in the log.
        num_bar (int, optional): The total number of steps. Defaults to 50.
    """
    print("=" * num_bar + log_name + "=" * num_bar)
    print(content)
    print("=" * (num_bar * 2 + len(log_name)))
