import pytest

from portable.megaminx_cluster.hardware_contract import publication_family


@pytest.mark.parametrize(("name", "sm", "family"), [("NVIDIA RTX 3070", 86, "Ampere"), ("NVIDIA RTX 4090", 89, "Ada"), ("NVIDIA L4", 89, "L4"), ("NVIDIA RTX PRO 6000 Blackwell", 120, "Blackwell"), ("NVIDIA H100 80GB HBM3", 90, "H100")])
def test_publication_family_matches_worker_inference(name, sm, family):
    assert publication_family(name, sm) == family


def test_publication_family_rejects_name_sm_mismatch():
    with pytest.raises(ValueError):
        publication_family("NVIDIA H100", 80)
