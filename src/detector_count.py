from pathlib import Path
import json
dir=Path(__file__).resolve().parent
file_path=dir/"data"/"processed"/"code_samples_100.jsonl"
file=file_path.open("r",encoding="utf-8")
content=file.read()
records=content.split("\n")
count=0
for record in records:
    if record:
        data = json.loads(record)
        if data["baseline_detected"]==1:
            count+=1
print(f"Baseline detected in {count} out of 100 samples.")
