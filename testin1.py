# Run this one-off debug script to inspect actual file content
import glob, os

HDD = 'E:'
all_inkml = glob.glob(os.path.join(HDD, 'inkcluster', '**', '*.inkml'), recursive=True)
print(f'Total files: {len(all_inkml)}')

# Print raw content of first 3 files
for fp in all_inkml[:3]:
    print(f'\n--- {fp} ---')
    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
        print(f.read()[:800])