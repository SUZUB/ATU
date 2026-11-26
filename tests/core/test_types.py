"""Unit conversions in core/types."""

import pytest

from atu2.core.types import Bps, bps_to_fraction


@pytest.mark.parametrize("bps,fraction", [(0.0, 0.0), (1.0, 0.0001), (2.0, 0.0002), (5, 0.0005), (10_000, 1.0)])
def test_bps_to_fraction(bps, fraction):
    assert bps_to_fraction(Bps(bps)) == pytest.approx(fraction)
