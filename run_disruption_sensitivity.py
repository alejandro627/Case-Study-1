from umdo_cpps import config
from umdo_cpps.disruption_sensitivity import export, run

def main():
    sensitivity_rows, summary_rows, report_lines = run()
    export(sensitivity_rows, summary_rows)

    print("=" * 80)
    print("Disruption sensitivity analysis (LHS) - one-by-one, then all together")
    print("=" * 80)
    for line in report_lines:
        print(line)
        print("-" * 80)
    print(f"CSV outputs written to: {config.OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
