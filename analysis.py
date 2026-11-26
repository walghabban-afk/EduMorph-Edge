#!/usr/bin/env python3
"""Deterministic synthetic companion analyses. All outputs are SIMULATION, not study evidence."""
from pathlib import Path
import json, argparse
import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'synthetic_results'
SEEDS={'classifier':20261124,'trial':20261125,'fairness':20261126}


def generate_classifier(seed=SEEDS['classifier'], n_students=240, windows_per_student=60):
    rng=np.random.default_rng(seed)
    n=n_students*windows_per_student
    labels=np.array(['Engaged']*(n//2+ n%2)+['Transitioning']*(n*3//10)+['Disengaged']*(n-(n//2+n%2)-n*3//10))
    rng.shuffle(labels)
    sid=np.repeat(np.arange(1,n_students+1),windows_per_student)
    rng.shuffle(sid)
    # features are explicitly synthetic proxies, not real sensor features
    df=pd.DataFrame({'student_id':sid,'window_index':np.tile(np.arange(windows_per_student),n_students),'label':labels,
      'response_latency_s':np.clip(rng.lognormal(1.0,.45,n),.1,20),
      'interaction_count':rng.poisson(4,n),'synthetic_signal':rng.normal(size=n)})
    counts=df.label.value_counts().reindex(['Engaged','Transitioning','Disengaged'],fill_value=0)
    return df,counts


def generate_trial(seed=SEEDS['trial'], n=300, schools=9):
    rng=np.random.default_rng(seed)
    ids=np.arange(1,n+1); school=np.tile(np.arange(1,schools+1),int(np.ceil(n/schools)))[:n]; rng.shuffle(school)
    treatment=np.zeros(n,dtype=int); treatment[rng.choice(n,n//2,replace=False)]=1
    gender=rng.choice(['F','M'],n); location=rng.choice(['Urban','Rural'],n,p=[.6,.4])
    # Synthetic scores: calibrated approximate means, intentionally not fitted to raw records.
    school_eff=rng.normal(0,2.0,schools)[school-1]
    baseline=rng.normal(61,4.5,n)
    post=baseline + 2 + treatment*18 + school_eff + rng.normal(0,4,n)
    # Set follow-up completion totals to reported approximate counts, random within arm.
    followup=np.zeros(n,dtype=int)
    for arm,complete in [(1,91),(0,87)]:
        ix=np.where(treatment==arm)[0]; followup[rng.choice(ix,complete,replace=False)]=1
    return pd.DataFrame({'participant_id':ids,'school_id':school,'treatment':treatment,'gender':gender,'location':location,
      'baseline_score':baseline,'post_score':post,'followup_complete':followup})


def generate_fairness(seed=SEEDS['fairness'],n=12000):
    rng=np.random.default_rng(seed)
    gender=rng.choice(['F','M'],n); location=rng.choice(['Urban','Rural'],n,p=[.6,.4])
    latent=rng.normal(size=n)+(gender=='M')*.05+(location=='Urban')*.03
    y=(latent+rng.normal(0,.7,n)>0).astype(int)
    pred=(latent+rng.normal(0,.55,n)>0.15).astype(int)
    return pd.DataFrame({'gender':gender,'location':location,'label':y,'prediction':pred})


def icc1(df, value='post_score',group='school_id'):
    groups=[g[value].dropna().values for _,g in df.groupby(group)]
    k=len(groups); n=sum(len(x) for x in groups); grand=np.mean([x.mean() for x in groups])
    ss_between=sum(len(x)*(x.mean()-grand)**2 for x in groups)
    ss_within=sum(((x-x.mean())**2).sum() for x in groups)
    msb=ss_between/(k-1); msw=ss_within/(n-k)
    avg_n=n/k
    return float((msb-msw)/(msb+(avg_n-1)*msw))


def attrition_stats(df):
    tab=pd.crosstab(df.treatment,df.followup_complete).reindex(index=[0,1],columns=[0,1],fill_value=0)
    chi2,p,dof,_=stats.chi2_contingency(tab,correction=False)
    return {'counts_treatment_control':{'treatment_n':int((df.treatment==1).sum()),'treatment_complete':int(tab.loc[1,1]),'control_n':int((df.treatment==0).sum()),'control_complete':int(tab.loc[0,1])},'chi2':float(chi2),'p':float(p),'dof':int(dof)}


def subgroup_effects(df):
    rows=[]
    for var in ['gender','location']:
      for level,g in df.groupby(var):
        a=g[g.treatment==1].post_score; b=g[g.treatment==0].post_score
        t,p=stats.ttest_ind(a,b,equal_var=False)
        rows.append({'group_variable':var,'group':level,'n_treatment':len(a),'n_control':len(b),'mean_treatment':a.mean(),'mean_control':b.mean(),'difference':a.mean()-b.mean(),'welch_t':t,'p_value':p})
    return pd.DataFrame(rows)


def fairness_metrics(df):
    rows=[]
    for group in ['gender','location']:
      groups=sorted(df[group].unique())
      vals=[]
      for lev,g in df.groupby(group):
        y=g.label.to_numpy(); pred=g.prediction.to_numpy()
        rate=pred.mean(); tpr=(pred[y==1].mean() if (y==1).sum() else np.nan); fpr=(pred[y==0].mean() if (y==0).sum() else np.nan)
        vals.append((lev,rate,tpr,fpr,(pred==y).mean()))
      for metric,idx in [('selection_rate',1),('TPR',2),('FPR',3),('accuracy',4)]:
        m=[v[idx] for v in vals]; gap=max(m)-min(m)
        for v in vals: rows.append({'group_variable':group,'group':v[0],'metric':metric,'value':v[idx],'max_min_gap':gap})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    classifier,counts=generate_classifier(); trial=generate_trial(); fairness=generate_fairness()
    classifier.to_csv(OUT/'synthetic_classifier_windows.csv',index=False); trial.to_csv(OUT/'synthetic_trial.csv',index=False); fairness.to_csv(OUT/'synthetic_fairness_predictions.csv',index=False)
    counts.rename_axis('class').to_csv(OUT/'classifier_class_counts.csv')
    subgroup_effects(trial).to_csv(OUT/'subgroup_effects.csv',index=False)
    fairness_metrics(fairness).to_csv(OUT/'fairness_metrics.csv',index=False)
    report={'classification_counts':{k:int(v) for k,v in counts.items()},'icc1_school_post_score':icc1(trial),
      'attrition':attrition_stats(trial),'seeds':SEEDS,
      'notice':'ALL DATA AND ANALYSES IN THIS REPORT ARE SYNTHETIC SIMULATIONS; NOT OBSERVED STUDY RESULTS.'}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2))
    (OUT/'SYNTHETIC_RESULTS_NOTICE.md').write_text('# Synthetic outputs notice\n\nEvery dataset and statistic in this folder is simulated with the fixed seeds documented in `summary.json`. These are software demonstrations and sensitivity illustrations, not observed participant results. Do not insert them into the manuscript as empirical findings.\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
