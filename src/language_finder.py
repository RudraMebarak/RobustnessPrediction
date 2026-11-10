from pathlib import Path
import csv
dir=Path(__file__).resolve().parent
csvfile=dir/"100_code_prompts_code_only.csv"
'''
languages = {"python": 0, "java": 0, "cpp": 0, "jsx": 0, "html": 0,"go": 0, "c":0, "php":0, "bash":0, "sql":0}
for i in range(100):
    file_path=dir/"data"/"raw"/"watermarked"/f"code_sample_{i+1}.txt"
    file=file_path.open("r",encoding="utf-8")
    content=file.read()
    lines=content.split("\n")
    flag=0
    for language in languages.keys():
        if language in lines[0]:
            languages[language]+=1
            file.close()
            flag=1
            break

    if "c++" in lines[0]:
        languages["cpp"]+=1
        file.close()
        flag=1

    if flag==0:
        print(f"code_sample_{i+1}.txt is not recognized")
        other+=1
print(languages)
print(f"Other: {other}")'''
with open(csvfile, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fields = reader.fieldnames

for i in range(100):
    file_path = dir/"data"/"raw"/"watermarked"/f"code_sample_{i+1}.txt"
    lines = file_path.read_text(encoding="utf-8").splitlines()

    language = None
    for lang in ["cpp", "python", "java", "jsx", "html", "go", "php", "bash", "sql", "c"]:
        if lang in lines[0].lower():
            language = lang
            break

    if "c++" in lines[0].lower():
        language = "cpp"

    if language:
        rows[i]["language_or_stack"] = language

with open(csvfile, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)