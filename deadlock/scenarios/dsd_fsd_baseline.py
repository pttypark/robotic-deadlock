from deadlock.dsd_fsd.experiment import run_comparison


def run(
    seed=0,
    max_steps=120,
    verbose=True,
    render=False,
    hold=False,
    save_dir=None,
    save_name=None,
    fps=6,
):
    if render:
        print("Render is not wired for the comparison runner; use the module CLI for one system.")
    result = run_comparison(seed=seed, max_steps=max_steps, stall_threshold=8)
    if verbose:
        print("DSD/FSD baseline comparison")
        print(result)
    return result
