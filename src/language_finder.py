from pathlib import Path
dir=Path(__file__).resolve().parent
other=0
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
print(f"Other: {other}")