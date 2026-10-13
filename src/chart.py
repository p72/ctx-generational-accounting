import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd
plt.rcParams["font.family"]="Noto Sans CJK JP"
c=pd.read_csv("../out_table2_compare.csv",index_col=0)
e=pd.read_csv("../out_delay_effect.csv",index_col=0)
fig,ax=plt.subplots(2,1,figsize=(7,8.5))
a=ax[0]
a.plot(c.index,c["論文_純負担"]/1000,"o-",label="佐藤（2019）表2",color="#1f77b4")
a.plot(c.index,c["再現_純負担"]/1000,"s--",label="再現（Python）",color="#d62728")
a.axhline(0,color="gray",lw=0.8)
a.set_title("世代別の生涯純負担（基本ケース、2010年時点の年齢）")
a.set_xlabel("2010年時点の年齢"); a.set_ylabel("百万円（2010年価値）"); a.legend(); a.grid(alpha=.3)
a.text(0.99,0.95,"将来世代の不均衡  論文 605.1% / 再現 605.5%",transform=a.transAxes,ha="right",va="top",fontsize=9)
b=ax[1]
b.bar(e.index-1,e["論文"],width=2,label="佐藤（2019）",color="#1f77b4")
b.bar(e.index+1,e["再現"],width=2,label="再現（Python）",color="#d62728")
b.set_title("2019年10月への延期による純負担の変化")
b.set_xlabel("2010年時点の年齢"); b.set_ylabel("千円"); b.legend(); b.grid(alpha=.3,axis="y")
plt.tight_layout(); plt.savefig("../ga_replication.png",dpi=160)
