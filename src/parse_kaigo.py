"""介護給付費実態調査（平成22年11月審査分）統計表3-1 → 年齢階級別・要介護度別 受給者数（千人）"""
import xlrd, pandas as pd
s = xlrd.open_workbook("../raw/kaigo10_toukei.xls").sheet_by_name("統計表3-1")
levels = ["要支援1", "要支援2", "要介護1", "要介護2", "要介護3", "要介護4", "要介護5"]
rows = []
for r in range(6, 14):                       # 総数ブロックの年齢階級行
    v = s.row_values(r)
    rows.append([str(v[2]).strip().replace("　", "")] + [float(x) for x in v[5:12]])
df = pd.DataFrame(rows, columns=["年齢階級"] + levels)
df.to_csv("../raw/kaigo2010_recipients.csv", index=False, encoding="utf-8-sig")
print(df)
