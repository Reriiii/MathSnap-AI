import csv
from pathlib import Path

def split_and_format_csv(input_csv="crohme23.csv", output_dir="dataset/processed"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    writers = {}
    files = {}
    
    for split in ["train", "val", "test"]:
        filepath = out_dir / f"{split}.csv"
        f = open(filepath, "w", encoding="utf-8", newline="")
        writer = csv.DictWriter(f, fieldnames=["image_path", "latex"])
        writer.writeheader()
        
        writers[split] = writer
        files[split] = f
        
    count = {"train": 0, "val": 0, "test": 0}
    
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            
            # Adjust split names to match our conventions
            if split == "valid":
                split = "val"
            
            if split not in writers:
                continue
                
            # Convert .\IMG\train\CROHME2013_train\form_001_E1.png 
            # to dataset/raw/IMG/train/CROHME2013_train/form_001_E1.png
            img_path = row["displayable_path"]
            if img_path.startswith(".\\"):
                img_path = img_path[2:]
            
            img_path_standardized = str(Path("dataset/raw") / Path(img_path))
            # replace backslashes
            img_path_standardized = img_path_standardized.replace("\\", "/")
            
            writers[split].writerow({
                "image_path": img_path_standardized,
                "latex": row["tokenized_label"]
            })
            count[split] += 1
            
    for f in files.values():
        f.close()
        
    print(f"Split complete: Train: {count['train']}, Val: {count['val']}, Test: {count['test']}")

if __name__ == "__main__":
    split_and_format_csv()
