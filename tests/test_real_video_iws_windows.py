"""Exact stored-index windows and internal-split boundaries, using synthetic arrays."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from shiftwm.real_video_iws.windows import IWSWindowDataset, eligible_starts, TASK_WIDTHS


def fixture_cache(task='pusht', lengths=(66,60,71)):
    width = TASK_WIDTHS[task]
    ids = ['000011','000012','000013']
    assignment = {ids[0]:'internal_train',ids[1]:'internal_train',ids[2]:'internal_development'}
    partitions = {s:sorted(k for k,v in assignment.items() if v==s) for s in ('internal_train','internal_development')}
    receipts={};arrays={};calls=[]
    for eid, n in zip(ids,lengths):
        receipts[eid]={'task':task,'episode_id':eid,'split':assignment[eid],'frames':n,'command_rows':n,
                       'command_width':width,'feature_dim':6144}
        features=np.arange(n,dtype=np.float32)[:,None]+np.arange(6144,dtype=np.float32)[None,:]/100000
        commands=np.arange(n,dtype=np.float32)[:,None]*100+np.arange(width,dtype=np.float32)[None,:]
        arrays[eid]={'features':features,'commands':commands,'frame_indices':np.arange(n,dtype=np.int64),
                     'command_row_indices':np.arange(n,dtype=np.int64)}
    def authorize(eid,split):
        if eid not in assignment or assignment[eid]!=split:raise ValueError('Reserved/wrong-split identity')
        return {'shapes':{'target_qpos':[receipts[eid]['frames'],width]}},assignment[eid]
    def episode(eid,split):
        authorize(eid,split);calls.append((eid,split));return receipts[eid],arrays[eid]
    inventory=SimpleNamespace(assignment=assignment,partitions=partitions,
        selected=lambda split:partitions[split],authorize=authorize)
    return SimpleNamespace(index=receipts,arrays=arrays,calls=calls,episode=episode,inventory=inventory,
        statistics={'fit_split':'internal_train','episodes':partitions['internal_train']})


@pytest.mark.parametrize('horizon',[2,15,30,45,60])
def test_unused_tail_rule_first_last_and_empty_windows(horizon):
    assert eligible_starts(horizon,horizon,5)==[]
    assert eligible_starts(horizon+1,horizon,5)==[0]
    assert eligible_starts(horizon+5,horizon,5)==[0]
    assert eligible_starts(horizon+6,horizon,5)==[0,5]
    assert eligible_starts(horizon+11,horizon,5)==[0,5,10]
    assert eligible_starts(1,horizon,5)==[]


@pytest.mark.parametrize('task',list(TASK_WIDTHS))
@pytest.mark.parametrize('horizon',[2,15,30,45,60])
def test_one_frame_H_commands_and_Hminus1_loss_targets(task,horizon):
    cache=fixture_cache(task,lengths=(horizon+6,horizon,horizon+11))
    data=IWSWindowDataset(cache,'internal_train',horizon=horizon,stride=5)
    assert data.windows==[(0,0),(0,5)]
    assert data.audit['ineligible_episodes']==1 and data.audit['records'][1]['windows']==0
    assert cache.calls==[('000011','internal_train'),('000012','internal_train')]
    for index,start in enumerate((0,5)):
        row=data[index];original=cache.arrays['000011']
        assert row['initial_features'].shape==(6144,)
        assert row['commands'].shape==(horizon,TASK_WIDTHS[task])
        assert row['targets'].shape==(horizon-1,6144)
        assert row['episode_id']=='000011' and row['window_start']==start and row['episode_index']==0
        np.testing.assert_array_equal(row['initial_features'],original['features'][start])
        np.testing.assert_array_equal(row['commands'],original['commands'][start:start+horizon])
        np.testing.assert_array_equal(row['targets'],original['features'][start+1:start+horizon])
        assert row['commands'][-1,0]==(start+horizon-1)*100
    batch=next(iter(torch.utils.data.DataLoader(data,batch_size=2)))
    assert batch['initial_features'].shape==(2,6144) and batch['episode_id']==['000011','000011']


def test_reserved_split_rejected_before_cache_or_payload_access():
    class Bomb:
        def __getattr__(self,key):raise AssertionError('Touched cache before rejecting split')
    for split in ('test','reserved_official_validation','internal_dev','val'):
        with pytest.raises(ValueError,match='reject reserved/test'):
            IWSWindowDataset(Bomb(),split)


def test_filter_and_authorize_every_identity_before_any_payload_read():
    cache=fixture_cache()
    cache.index['000012']['split']='reserved_official_validation'
    with pytest.raises(ValueError,match='split identity'):
        IWSWindowDataset(cache,'internal_train')
    assert cache.calls==[]
    cache=fixture_cache()
    cache.index['000000']={**cache.index['000011'],'episode_id':'000000'}
    with pytest.raises(ValueError,match='reserved identities'):
        IWSWindowDataset(cache,'internal_train')
    assert cache.calls==[]


def test_development_reads_only_dev_and_keeps_training_statistics():
    cache=fixture_cache()
    data=IWSWindowDataset(cache,'internal_development')
    assert cache.calls==[('000013','internal_development')]
    assert data.episode_ids==['000013'] and data.windows==[(0,0),(0,5),(0,10)]
    assert data.statistics['episodes']==['000011','000012']


def test_all_short_trajectories_retain_audit_without_fabricated_query():
    data=IWSWindowDataset(fixture_cache(lengths=(60,59,71)),'internal_train')
    assert len(data)==0 and data.audit['ineligible_episodes']==2
    assert len(data.audit['records'])==2
    with pytest.raises(IndexError):data[0]


@pytest.mark.parametrize('mutation', ['command_tail','native_index','nonfinite','wrong_stats'])
def test_command_layout_row_indices_and_training_stats_fail_closed(mutation):
    cache=fixture_cache()
    if mutation=='command_tail':cache.arrays['000011']['commands']=cache.arrays['000011']['commands'][:-1]
    elif mutation=='native_index':cache.arrays['000011']['frame_indices'][3]+=1
    elif mutation=='nonfinite':cache.arrays['000011']['features'][0,0]=np.nan
    else:cache.statistics['fit_split']='internal_development'
    with pytest.raises(ValueError):IWSWindowDataset(cache,'internal_train')


@pytest.mark.parametrize('args',[(60,1,5),(60,60,0),(60,60,1.5),(-1,60,5),(60,True,5)])
def test_invalid_window_arguments_rejected(args):
    with pytest.raises(ValueError):eligible_starts(*args)
