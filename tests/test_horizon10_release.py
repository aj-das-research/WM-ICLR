from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('h10_publisher',ROOT/'scripts/publishing/publish_horizon10_release.py')
publisher=importlib.util.module_from_spec(spec);spec.loader.exec_module(publisher)


@pytest.fixture
def proof():
    return {'status':'passed','network_attempts':0,'physical_relocation':True,'device':'cpu',
        'models':[{'run':name,'epoch':1,'maximum_absolute_difference':0.,'shape':[1,10,1536]}
                  for name in sorted(publisher.EXPECTED)]}


def test_complete_proof_passes(proof):
    publisher.verify_proof(proof)


@pytest.mark.parametrize('damage',['missing','duplicate','network','not_relocated','gpu','wrong_epoch','approximate','wrong_shape'])
def test_incomplete_or_changed_parity_rejected(proof,damage):
    if damage=='missing':proof['models'].pop()
    elif damage=='duplicate':proof['models'][-1]=deepcopy(proof['models'][0])
    elif damage=='network':proof['network_attempts']=1
    elif damage=='not_relocated':proof['physical_relocation']=False
    elif damage=='gpu':proof['device']='cuda'
    elif damage=='wrong_epoch':proof['models'][0]['epoch']=30
    elif damage=='approximate':proof['models'][0]['maximum_absolute_difference']=1e-8
    else:proof['models'][0]['shape']=[1,5,1536]
    with pytest.raises(ValueError):publisher.verify_proof(proof)


def test_readme_keeps_mixed_selection_and_distinct_loader():
    text=publisher.readme('ALL TWENTY REGISTERED COMPARISONS')
    assert 'epoch 1' in text and '30 epochs' in text
    assert 'horizon10_train import load_package' in text
    assert '-0.116%' in text and 'interval including zero' in text
    assert 'ALL TWENTY REGISTERED COMPARISONS' in text
    assert 'h5 standard population and h5 prefixes' in text
