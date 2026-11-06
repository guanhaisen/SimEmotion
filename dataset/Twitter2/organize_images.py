#!/usr/bin/env python3
"""
Script to organize Twitter2 dataset images by sentiment labels.
Copies images from partition folders to pos/neg directories based on groundTruthLabel.txt.
"""

import os
import shutil
from pathlib import Path

def organize_images():
    # Setup paths
    base_dir = Path(r"D:\DatasetForSentiment\Twitter2")
    label_file = base_dir / "groundTruthLabel.txt"
    pos_dir = base_dir / "pos"
    neg_dir = base_dir / "neg"
    partition_dirs = [base_dir / f"partition{i}" for i in range(1, 6)]

    # Create destination directories
    pos_dir.mkdir(exist_ok=True)
    neg_dir.mkdir(exist_ok=True)

    # Read ground truth labels
    labels = {}
    with open(label_file, 'r') as f:
        # Skip header
        next(f)
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    image_name, sentiment = parts
                    labels[image_name] = sentiment

    print(f"Loaded {len(labels)} labels")
    print(f"Positive: {sum(1 for s in labels.values() if s == 'positive')}")
    print(f"Negative: {sum(1 for s in labels.values() if s == 'negative')}")

    # Counters for moved files
    pos_count = 0
    neg_count = 0
    not_found = 0

    # Process each image
    for image_name, sentiment in labels.items():
        # Search for image in partition directories
        image_path = None
        for partition_dir in partition_dirs:
            potential_path = partition_dir / image_name
            if potential_path.exists():
                image_path = potential_path
                break

        if image_path is None:
            print(f"Warning: {image_name} not found in any partition")
            not_found += 1
            continue

        # Copy to appropriate directory
        if sentiment == "positive":
            dest_path = pos_dir / image_name
            shutil.copy2(image_path, dest_path)
            pos_count += 1
        elif sentiment == "negative":
            dest_path = neg_dir / image_name
            shutil.copy2(image_path, dest_path)
            neg_count += 1
        else:
            print(f"Warning: Unknown sentiment '{sentiment}' for {image_name}")

    print(f"\nOrganization complete:")
    print(f"Positive images copied: {pos_count}")
    print(f"Negative images copied: {neg_count}")
    print(f"Images not found: {not_found}")
    print(f"Total processed: {pos_count + neg_count}")

if __name__ == "__main__":
    organize_images()