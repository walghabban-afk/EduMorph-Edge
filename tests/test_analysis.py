import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from synthetic_analysis import generate_classifier, generate_trial, generate_fairness, icc1, attrition_stats, subgroup_effects, fairness_metrics

def test_classifier_reproducible_and_counts():
    a,c1=generate_classifier(); b,c2=generate_classifier()
    assert a.equals(b) and c1.equals(c2)
    assert c1.to_dict()=={'Engaged':7200,'Transitioning':4320,'Disengaged':2880}

def test_trial_reproducible_and_arm_completion_counts():
    a=generate_trial(); b=generate_trial(); assert a.equals(b)
    assert int(a[a.treatment==1].followup_complete.sum())==91
    assert int(a[a.treatment==0].followup_complete.sum())==87

def test_synthetic_functions():
    d=generate_trial(); assert -1<=icc1(d)<=1
    assert 'p' in attrition_stats(d)
    assert len(subgroup_effects(d))==4
    assert not fairness_metrics(generate_fairness()).empty
