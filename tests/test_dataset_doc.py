from firespread import dataset


def test_every_layer_and_number_is_documented():
    missing = [k for k in dataset.SPATIAL + dataset.SCALARS if k not in dataset.DOC]
    assert not missing, missing
    assert len(set(dataset.SPATIAL)) == len(dataset.SPATIAL) and len(set(dataset.SCALARS)) == len(dataset.SCALARS)
