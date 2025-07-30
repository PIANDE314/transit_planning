from pathlib import Path
from transitlib.accessibility.compare import calculate_accessibility

# Directory containing your 72 GTFS folders
gtfs_root = Path(".")

# Output directory for results
outputs_dir = Path("output")
outputs_dir.mkdir(parents=True, exist_ok=True)

# Find all GTFS folders matching the pattern *_*_*_*_*_gtfs
gtfs_dirs = sorted(gtfs_root.glob("*_*_*_*_*_gtfs"))

results = []

for gtfs_path in gtfs_dirs:
    name = gtfs_path.name
    try:
        score = calculate_accessibility(str(gtfs_path))
    except Exception as e:
        print(f"❌ {name}: ERROR ({e})")
        continue

    # Print to console
    print(f"{name}: {score:.4f}")

    # Save to a text file
    out_file = outputs_dir / f"{name}_accessibility.txt"
    with out_file.open("w") as f:
        f.write(f"{name}: {score:.6f}\n")

    results.append((name, score))

# Optionally, write a summary CSV
import csv
with (outputs_dir / "accessibility_summary.csv").open("w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["gtfs_folder", "accessibility_score"])
    for name, score in results:
        writer.writerow([name, f"{score:.6f}"])
