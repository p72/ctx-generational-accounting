import numpy as np, pandas as pd, ga_model as m, paper
P=m.load_population(); import sys; prof=m.load_profiles(sys.argv[1] if len(sys.argv)>1 else "head"); H=m.per_capita_profiles(P.loc[m.BASE].values, prof)
# 検算：配分後の総額がSNA総額に一致するか
chk = (H.mul(P.loc[2010].values, axis=0).sum()/1e9).round(1)
tot = {n: s*t for n,t,_,s in m.ITEMS}
assert all(abs(chk[k]-tot[k])<0.5 for k in tot), "allocation mismatch"
print("配分検算OK（全18項目でSNA総額と一致）")
res={}
for sc in ["base","delay","keep8"]:
    t,r=m.table(sc,P,H); res[sc]=(t,r)
t2=paper.table2()
cmp=pd.DataFrame({"論文_負担":t2.burden,"再現_負担":res['base'][0]["負担"],
                  "論文_受益":t2.benefit,"再現_受益":res['base'][0]["受益"],
                  "論文_純負担":t2.net,"再現_純負担":res['base'][0]["純負担"]})
pd.set_option("display.width",200); print(cmp.round(0).to_string())
print()
rows=[]
for sc in ["base","delay","keep8"]:
    r=res[sc][1]; s=paper.SUMMARY[sc]
    rows.append([sc, s['debt'], r['implicit_debt']/1e12, s['future'], r['future']/1e3, s['imb'], r['imbalance']*100])
summ=pd.DataFrame(rows,columns=["ケース","論文_潜在債務(兆円)","再現_潜在債務","論文_将来世代(千円)","再現_将来世代","論文_不均衡%","再現_不均衡%"])
print(summ.round(1).to_string(index=False))
# 延期の効果（delay − base）
d_rep=(res['delay'][0]["純負担"]-res['base'][0]["純負担"])
d_pap=pd.Series(np.array(paper.T3_net)-np.array(paper.T2['net']),index=paper.AGES5)
eff=pd.DataFrame({"論文":d_pap,"再現":d_rep}).round(1)
print("\n延期による純負担の変化（千円）\n",eff.T.to_string())
print("将来世代: 論文 +%.0f千円 / 再現 +%.0f千円"%(paper.SUMMARY['delay']['future']-paper.SUMMARY['base']['future'],(res['delay'][1]['future']-res['base'][1]['future'])/1e3))
print("潜在債務: 論文 +%.1f兆円 / 再現 +%.1f兆円"%(1483.2-1472.5,(res['delay'][1]['implicit_debt']-res['base'][1]['implicit_debt'])/1e12))
print("8%%据え置き 潜在債務: 論文 +%.1f兆円 / 再現 +%.1f兆円"%(1579.7-1472.5,(res['keep8'][1]['implicit_debt']-res['base'][1]['implicit_debt'])/1e12))
cmp.to_csv("../out_table2_compare.csv",encoding="utf-8-sig"); summ.to_csv("../out_summary.csv",index=False,encoding="utf-8-sig"); eff.to_csv("../out_delay_effect.csv",encoding="utf-8-sig")
