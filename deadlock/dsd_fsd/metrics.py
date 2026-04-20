def summarize_metrics(ledger, controller_stats, steps):
    completed = ledger.completed()
    flow_times = [tx.flow_time() for tx in completed if tx.flow_time() is not None]
    waiting_times = [tx.waiting_time(tx.completed_step or steps) for tx in completed]
    f_avg = sum(flow_times) / len(flow_times) if flow_times else 0.0
    f_max = max(flow_times) if flow_times else 0
    wait_avg = sum(waiting_times) / len(waiting_times) if waiting_times else 0.0
    wr = wait_avg / f_avg if f_avg else 0.0

    metrics = {
        "steps": steps,
        "created": len(ledger.transactions),
        "completed": len(completed),
        "f_avg": f_avg,
        "f_max": f_max,
        "wr": wr,
        "avg_wait": wait_avg,
        **controller_stats,
    }
    return {key: _plain_number(value) for key, value in metrics.items()}


def _plain_number(value):
    if hasattr(value, "item"):
        return value.item()
    return value
