from umdo_cpps import config
from umdo_cpps.lhs_convergence import METRICS, export, plot, recommend_n, run, suffix_for_current_scenario

def main():
    suffix = suffix_for_current_scenario()
    raw_rows, summary_rows = run()
    export(raw_rows, summary_rows, suffix)

    out_path = config.OUTPUT_DIR / f"lhs_convergence_study_{suffix}.png"
    plot(summary_rows, out_path, f"LHS sample-size convergence - {suffix}")

    print("=" * 80)
    print("LHS convergence study")
    print("=" * 80)
    print(f"Sample sizes tested: {[row['N'] for row in summary_rows]}")
    print(f"Replications per size: {summary_rows[0]['n_replications']}")
    print()
    header = f"{'N':>6s}" + "".join(f"{m:>28s}" for m in METRICS)
    print(header)
    for row in sorted(summary_rows, key=lambda r: r["N"]):
        line = f"{row['N']:>6d}"
        for m in METRICS:
            line += f"  {row[f'{m}_estimate']:>10.4f} +/-{row[f'{m}_sem_across_replications']:>7.4f} ({100*row[f'{m}_relative_sem']:>5.1f}%)"
        print(line)
    print()
    for m in METRICS:
        n_ok = recommend_n(summary_rows, m)
        print(f"Smallest N with relative SEM <= 2% for {m}: {n_ok if n_ok is not None else 'not reached within tested range'}")
    print()
    print(f"CSV/plot outputs written to: {config.OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
