from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence
import csv
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

HEADER_FILL = PatternFill("solid", fgColor="0F766E")
HEADER_FONT = Font(color="FFFFFF", bold=True)
TITLE_FILL = PatternFill("solid", fgColor="C65911")
TITLE_FONT = Font(color="FFFFFF", bold=True, size=14)

def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]):
    if not rows: return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def add_sheet(wb: Workbook, name: str, rows: Sequence[Mapping[str, Any]]):
    ws=wb.create_sheet(title=name[:31])
    if not rows:
        ws["A1"]="No data"; return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    for c,k in enumerate(fields,1):
        cell=ws.cell(1,c,k); cell.fill=HEADER_FILL; cell.font=HEADER_FONT
        cell.alignment=Alignment(horizontal="center",vertical="center",wrap_text=True)
    for i,r in enumerate(rows,2):
        for c,k in enumerate(fields,1):
            v=r.get(k)
            if isinstance(v,(str,int,float,bool)) or v is None:
                ws.cell(i,c,v)
            else:
                ws.cell(i,c,str(v))
    ws.freeze_panes="A2"
    for c,k in enumerate(fields,1):
        width=min(42,max(12,len(str(k))+2))
        for i in range(2,min(ws.max_row,100)+1):
            width=max(width,min(42,len(str(ws.cell(i,c).value or ""))+2))
        ws.column_dimensions[get_column_letter(c)].width=width

def export_workbook(path: Path, sheets: Mapping[str, Sequence[Mapping[str, Any]]]):
    wb=Workbook()
    wb.remove(wb.active)
    for name,rows in sheets.items():
        add_sheet(wb,name,rows)
    wb.save(path)
