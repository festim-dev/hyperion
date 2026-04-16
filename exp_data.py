# exp_data.py
# Experimental flux data for all runs.
# flux_err is k=2 (95% confidence interval).

# Experimental cases: (T [C], run label, P_up [Pa], P_down [Pa], flux [H/s], flux_err [H/s])
dry_run = [
    (500.0, "Run 1", 1.30e5, 1.98e2, 4.52e16, 3.06e15),
    (500.0, "Run 2", 1.10e5, 1.79e2, 4.08e16, 2.76e15),
    (600.0, "Run 1", 1.30e5, 4.59e2, 1.03e17, 6.97e15),
    (600.0, "Run 2", 1.10e5, 4.02e2, 9.19e16, 6.22e15),
    (700.0, "Run 1", 1.30e5, 8.16e2, 1.85e17, 1.25e16),
    (700.0, "Run 2", 1.10e5, 7.36e2, 1.65e17, 1.12e16),
]
