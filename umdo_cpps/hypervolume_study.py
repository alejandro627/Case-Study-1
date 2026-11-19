from __future__ import annotations
from . import config
from .exporters import write_csv
from .optimisation import hypervolume_history, run_nsga2
from .plots import hypervolume_convergence
from .scenarios import get_scenario

def run():
    """Re-run NSGA-II with save_history=True (not part of the main workflow) and write the hypervolume-vs-generation
        convergence plot and CSV.
    """
    scenario = get_scenario(config.SCENARIO_MODE)
    res = run_nsga2(save_history=True)
    rows, ref_point = hypervolume_history(res)

    suffix = f"scenario_{config.SCENARIO_MODE}_{scenario['name'].lower()}"
    out = config.OUTPUT_DIR
    write_csv(out / f"hypervolume_history_{suffix}.csv", rows)
    hypervolume_convergence(
        rows, out / f"hypervolume_convergence_{suffix}.png",
        f"Hypervolume convergence - {scenario['name']} "
        f"(ref_point=[{ref_point[0]:.3f}, {ref_point[1]:.3f}, {ref_point[2]:.3f}])",
    )

    print("=" * 80)
    print("Hypervolume study completed")
    print(f"Reference point (F-space, [-TH, E, -R_phi]): {list(ref_point)}")
    print(f"Generations tracked: {len(rows)}")
    print(f"Final hypervolume (gen {rows[-1]['n_gen']}): {rows[-1]['hypervolume']:.4f}")
    print(f"Outputs: {out.resolve()}")
    return rows, ref_point

if __name__ == "__main__":
    run()
