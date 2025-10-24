import os

root_dir = "/home/anurizada/Documents/bar_linkages"

# Dictionary to store folder -> file count
folder_counts = {}

for dirpath, dirnames, filenames in os.walk(root_dir):
    # Count total files recursively in this subfolder
    total_files = sum(len(files) for _, _, files in os.walk(dirpath))
    folder_counts[dirpath] = total_files

# Sort by number of files (descending)
sorted_folders = sorted(folder_counts.items(), key=lambda x: x[1], reverse=True)

# Print results
print(f"{'Folder':<80} | {'# Files':>7}")
print("-" * 90)
for folder, count in sorted_folders:
    print(f"{folder:<80} | {count:>7}")
