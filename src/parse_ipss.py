import xlrd, re, pandas as pd
rows=[]
for f in ['../raw/ipss_1-9.xls','../raw/ipss_s1-9.xls']:
    b=xlrd.open_workbook(f)
    for s in b.sheets():
        m=re.search(r'\((\d{4})\)', s.cell_value(1,0)); y=int(m.group(1))
        for r in range(4,s.nrows):
            for c in (0,5):
                a=s.cell_value(r,c); v=s.cell_value(r,c+1)
                if a=='' or v=='': continue
                age=105 if isinstance(a,str) else int(a)
                rows.append((y,age,float(v)*1000))
df=pd.DataFrame(rows,columns=['year','age','pop']).groupby(['year','age'])['pop'].sum().unstack()
df.to_csv('../raw/ipss_pop.csv'); print(df.shape, df.index.min(), df.index.max(), df.sum(axis=1).loc[[2015,2065,2115]])
