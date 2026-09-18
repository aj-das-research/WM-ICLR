"""Prespecified episode selection, independent of rendering and measured scores."""
import math
RULE = {
    'scope':'original_validation_only',
    'modes':['autoregressive','transport'],
    'seeds':[0,1,2],
    'metric':'native_mse_at_query_step_10',
    'episode_aggregation':'mean complete windows within each episode, then equal mean of three seeds',
    'ranking':'100*(autoregressive_episode_mse-transport_episode_mse)/autoregressive_episode_mse',
    'tie_break':'ascending episode_id',
    'selected_ranks':'descending_gain ranks 0, floor((N-1)/2), N-1',
    'roles':['largest_episode_gain','median_episode_gain','smallest_episode_gain'],
    'regression_label':'use largest regression only when minimum episode gain is strictly negative',
    'replay_window':'smallest eligible h10 window_start; never reselect for visual advantage',
    'target_patch_zero_based':[1,1],
    'frames':'three observed support frames and tenth-query target, at recorded native frame indices',
    'error_map':'mean squared standardized error per patch over 384 channels, then mean of three seeds',
    'error_scale':'linear zero to maximum of all six displayed endpoint maps, no percentile clipping',
    'transport_panel':'mean three-seed h10 row of transport matrix for fixed target patch (1,1); show its gate separately',
    'transport_scale':[0,1],
    'replay_tolerance':{'rtol':2e-5,'atol':2e-6},
    'publication_gate':'candidate only; completed verified 15-model campaign plus numeric checks plus actual visual review',
}


def select_episodes(rows):
    if len(rows)<3 or len({r['episode_id'] for r in rows})!=len(rows):
        raise ValueError('Three or more unique matched validation episodes required')
    for row in rows:
        if not math.isfinite(row['gain_percent']):raise ValueError('Nonfinite episode gain')
    ranked=sorted(rows,key=lambda row:(-row['gain_percent'],row['episode_id']))
    positions=[0,(len(ranked)-1)//2,len(ranked)-1]
    selected=[]
    for role,rank in zip(RULE['roles'],positions):
        row=ranked[rank]
        label={'largest_episode_gain':'Largest episode gain','median_episode_gain':'Median episode gain',
               'smallest_episode_gain':'Largest regression' if row['gain_percent']<0 else 'Smallest episode gain'}[role]
        selected.append({**row,'role':role,'label':label,'rank_descending':rank,'population_size':len(ranked)})
    return selected
