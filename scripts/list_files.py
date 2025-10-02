from pathlib import Path
from datetime import datetime
import pandas as pd
import os

root=r"D:\!!ChinaDATA\z_reference\Papers"    
allowed_extensions=None

root = Path(root)
rows = []

# Walk recursively; handle long paths & non-ASCII safely via pathlib
for p in root.rglob("*"):
    # skip directories
    if not p.is_file():
        continue

    # skip temp/hidden junk (customize as needed)
    name = p.name
    if name.startswith("~$") or name.endswith(".tmp"):
        continue

    # filetype: extension without leading dot, lowercase; empty string if none
    ext = p.suffix.lower().lstrip(".")

    # optional filtering
    if allowed_extensions is not None and ext not in allowed_extensions:
        continue

    try:
        # last modified time → YYYYMMDD
        ts = p.stat().st_mtime  # seconds since epoch
        dt = datetime.fromtimestamp(ts)
        date_modified = dt.strftime("%Y%m%d")
    except OSError:
        # in case of permission or race-condition errors
        date_modified = ""

    rows.append({
        "file": str(p),          # full path
        "filetype": ext,         # e.g., "pdf"
        "date_modified": date_modified
    })

df = pd.DataFrame(rows, columns=["file", "filetype", "date_modified"])

# (Optional) stable sort by path then date desc
# df = df.sort_values(["file", "date_modified"], ascending=[True, False], ignore_index=True)

# --- Examples ---
# 1) All files
# df_all = collect_papers_info()

# 2) Only PDFs (recommended for your example)
# df_pdf = collect_papers_info(allowed_extensions={"pdf"})
# print(df_pdf.head())


df = df.sort_values(["filetype","file", "date_modified"], ascending=[True, True, False], ignore_index=True)
df.to_csv("D://papers_list.csv", index=False)
os.system("start D://papers_list.csv")

# export df to csv with utf-8 encoding
df.to_csv("D://papers_list.csv", index=False, encoding="utf-8")
os.system("start D://papers_list.csv")