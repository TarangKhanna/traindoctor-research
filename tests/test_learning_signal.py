"""Advantage invariants: aggregate disagreement alone cannot identify signal."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('learning_signal',Path(__file__).parents[1]/'scripts/audit_learning_signal.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

def group(v1,v2):
    return [dict(policy='esft_full',episode=0,sample=i,kind='A1',role='decoy',failure='no_completed_relaunch',v0=1,v1=x,v2=y,repair_ok=bool(y)) for i,(x,y) in enumerate(zip(v1,v2))]

def test_constant_shift_changes_rewards_but_not_advantages():
    s,_=audit.summarize(group([1]*8,[0]*8))
    assert s['disagreement']==8 and s['normalized_different']==0

def test_single_changed_reward_changes_entire_group_centering():
    x=[1,1,0,0,0,0,0,0];y=[1,0,0,0,0,0,0,0]
    s,_=audit.summarize(group(x,y))
    assert s['normalized_different']==1
    assert not np.any(np.isclose(audit.advantages(x,False),audit.advantages(y,False)))

def test_incomplete_group_is_not_invented():
    with pytest.raises(AssertionError):audit.summarize(group([1]*4,[0]*4))
