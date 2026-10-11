import numpy as np
import pytest
from transport_reasoning.reachability.metrics import binary_metrics,clustered_ba_interval
from transport_reasoning.reachability.campaign import choose_admission
from transport_reasoning.reachability.config import PRIORITY,FAMILIES


def test_metric_ties_single_class_and_threshold():
    x=binary_metrics([0,0,1,1],[-2,0,0,2]);assert x['tp']==2 and x['tn']==1 and x['balanced_accuracy']==.75
    assert x['auroc']==.875 and x['average_precision']==pytest.approx(5/6)
    assert binary_metrics([1,1],[0.,1.])['balanced_accuracy'] is None
    assert binary_metrics([1,1],[0.,1.])['auroc'] is None
    assert binary_metrics([],[])['count']==0
    perfect=binary_metrics([0,1],[-10,10]);assert perfect['balanced_accuracy']==perfect['auroc']==1


def test_bootstrap_keeps_queries_in_graphs():
    result=clustered_ba_interval([1,0,1,0],[1,-1,1,-1],['a','a','b','b'],repeats=30)
    assert result['clusters']==2 and result['ba_lower95']==result['ba_upper95']==1


def test_budget_selection_keeps_controls_families_and_seed_rounds():
    rows=[{'family':f,'model':m,'pilot_seconds_estimate':10.,'final_including_evaluation_seconds_estimate':100.} for m in PRIORITY for f in FAMILIES]
    profile={'rows':rows,'full_grid_with_reserve_seconds':11448.}
    tiny=choose_admission(profile,100);assert tiny['status']=='budget_stopped'
    reduced=choose_admission(profile,1000);assert reduced['families']==FAMILIES and reduced['models']==PRIORITY[:2] and reduced['seeds']==[11]
    assert reduced['required_with_reserve_seconds']<=1000
    full=choose_admission(profile,20000);assert full['full_grid_admitted']
