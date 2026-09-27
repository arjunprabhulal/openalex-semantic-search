import pytest

from openalex_semantic_search.config import STAGE_ORDER, get_stage, require_stage_confirmation


def test_stages_are_ordered_and_guarded():
    counts = [get_stage(name).records for name in STAGE_ORDER]
    assert counts == sorted(counts)
    require_stage_confirmation(get_stage("10k"), None)
    with pytest.raises(ValueError, match="BUILD_100K"):
        require_stage_confirmation(get_stage("100k"), None)
    require_stage_confirmation(get_stage("100k"), "BUILD_100K")
    assert get_stage("1m").nprobe == 192
    assert get_stage("full").nprobe == 192


def test_unknown_stage_lists_choices():
    with pytest.raises(ValueError, match="10k, 100k"):
        get_stage("infinite")
