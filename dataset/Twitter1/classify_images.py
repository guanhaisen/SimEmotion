#!/usr/bin/env python3
import csv
import os
import shutil

def classify_images():
    # Define paths
    base_dir = "D:\DatasetForSentiment\Twitter1"
    csv_file = os.path.join(base_dir, "amt_result.csv")
    source_dir = os.path.join(base_dir, "Agg_AMT_Candidates")
    pos_dir = os.path.join(base_dir, "pos")
    neg_dir = os.path.join(base_dir, "neg")

    # Create target directories if they don't exist
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    # Track statistics
    pos_count = 0
    neg_count = 0
    not_found = 0

    # Read and process the CSV file
    with open(csv_file, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)

        # Skip header
        header = next(csv_reader)

        # Process each row
        for row in csv_reader:
            if len(row) < 4:
                continue

            img_name = row[0].strip()
            votes_neg = int(row[2])
            votes_pos = int(row[3])

            # Determine classification
            if votes_neg > votes_pos:
                target_dir = neg_dir
                neg_count += 1
                classification = "negative"
            elif votes_pos > votes_neg:
                target_dir = pos_dir
                pos_count += 1
                classification = "positive"
            else:
                # Handle tie case - skip or put in one folder based on preference
                print(f"Tie for {img_name}: neg={votes_neg}, pos={votes_pos}")
                continue

            # Source and destination paths
            source_path = os.path.join(source_dir, img_name)
            target_path = os.path.join(target_dir, img_name)

            # Move file if it exists
            if os.path.exists(source_path):
                shutil.move(source_path, target_path)
                print(f"Moved {img_name} to {classification} folder")
            else:
                not_found += 1
                print(f"Image not found: {img_name}")

    # Print summary
    print(f"\nClassification complete!")
    print(f"Positive images: {pos_count}")
    print(f"Negative images: {neg_count}")
    print(f"Images not found: {not_found}")
    print(f"Total processed: {pos_count + neg_count}")

if __name__ == "__main__":
    classify_images()