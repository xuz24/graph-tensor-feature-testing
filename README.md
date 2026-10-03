# graph-tensor-feature-testing

## Log tables

Run `python3 scripts/logs_to_markdown.py logs` to write one Markdown report per
run directory to `logs/markdown/`. Each run must contain `0.out` through `11.out`,
covering folds 0–3 and seeds 0–2 for one setup. Setup, fold, and seed are read
from the log contents. All arms in the final results table are included.

For a single run or a custom output directory:

```sh
python3 scripts/logs_to_markdown.py logs/drug_feature_10arm_61617210 --output-dir logs/tables
```

Reports contain per-seed results, per-fold averages, per-seed averages across
folds, and the grand average. Metrics use three decimal places; query counts
use integers for individual results and one decimal place for averages.
Averages are unweighted arithmetic means of the values printed in the logs
(already rounded by training); NaN values propagate. The extra `mrr` column
is omitted. Invalid or incomplete runs are reported and skipped, with a nonzero
exit status; other valid runs are still converted. Existing reports for valid
runs are overwritten when rerunning the command.
