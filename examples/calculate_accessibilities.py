from pathlib import Path
import itertools
import csv

from transitlib.accessibility.compare import calculate_accessibility

# 1) Define the parameter grid
cities     = ["navi_mumbai", "chennai", "maputo"]
noise      = ["clean", "noisy"]
trip_type  = ["uni", "gam", "norm"]
transform  = ["lin", "sqrt"]
algorithm  = ["HC", "SA"]

# 2) Output directory
outputs_dir = Path("output")
outputs_dir.mkdir(parents=True, exist_ok=True)

# 3) Collect results
results = []

# 4) Loop over every combination, build the GTFS folder name, compute accessibility
for city, n, t, tf, alg in itertools.product(cities, noise, trip_type, transform, algorithm):
    folder_name = f"{city}_{n}_{t}_{tf}_{alg}_gtfs"
    gtfs_path = outputs_dir / folder_name
    if not gtfs_path.exists():
        print(f"⚠️ Skipping missing folder: {folder_name}")
        continue

    try:
        score = calculate_accessibility(str(gtfs_path))
    except Exception as e:
        print(f"❌ Error processing {folder_name}: {e}")
        continue

    # Print to console
    print(f"{folder_name}: {score:.4f}")

# 5) Write summary CSV
summary_csv = outputs_dir / "accessibility_summary.csv"
with summary_csv.open("w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["gtfs_folder", "accessibility_score"])
    for name, score in results:
        writer.writerow([name, f"{score:.6f}"])

print(f"\n✅ Completed! Results written to '{outputs_dir.absolute()}'")
