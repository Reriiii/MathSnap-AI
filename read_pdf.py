import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pypdf import PdfReader
r = PdfReader(r'd:\Workplace\hmer\2407.11380v1.pdf')

pages_of_interest = list(range(4, 14)) + [18, 19, 20, 21]  # 0-indexed

for i in pages_of_interest:
    if i < len(r.pages):
        txt = r.pages[i].extract_text() or ''
        print(f'\n{"="*60}\n PAGE {i+1}\n{"="*60}')
        print(txt)
