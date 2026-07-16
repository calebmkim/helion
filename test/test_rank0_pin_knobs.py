from __future__ import annotations

import unittest
from unittest.mock import patch

from helion._compiler.cute.tcgen05_config import _config_violates_rank0_pin
from helion._compiler.cute.tcgen05_config import _pin_enum_search
from helion._compiler.cute.tcgen05_config import _rank0_pinned_enum_values
from helion._compiler.cute.tcgen05_config import rank0_pin_knobs_enabled
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_fragment import ListOf


class TestRank0PinKnobs(unittest.TestCase):
    """Unit tests for the ``HELION_RANK0_PIN_KNOBS`` experiment arm helpers
    (``experiments/cute_knob_restriction``). Pure-Python; no GPU/codegen needed.
    """

    def test_flag_off_by_default(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("HELION_RANK0_PIN_KNOBS", None)
            self.assertFalse(rank0_pin_knobs_enabled())

    def test_flag_reads_env(self) -> None:
        with patch.dict("os.environ", {"HELION_RANK0_PIN_KNOBS": "1"}):
            self.assertTrue(rank0_pin_knobs_enabled())
        with patch.dict("os.environ", {"HELION_RANK0_PIN_KNOBS": "0"}):
            self.assertFalse(rank0_pin_knobs_enabled())

    def test_pin_enum_narrows_search_not_choices(self) -> None:
        # search_choices collapses to the pin; the validation-facing `choices`
        # stays full, so an explicit user config still validates.
        frag = EnumFragment(choices=(2, 4))
        pinned = _pin_enum_search(frag, 2)
        self.assertEqual(pinned.search_choices, (2,))
        self.assertEqual(pinned.choices, (2, 4))
        # random() only ever draws the pinned value now
        self.assertEqual({pinned.random() for _ in range(50)}, {2})

    def test_pin_enum_rejects_out_of_range_value(self) -> None:
        frag = EnumFragment(choices=("role_local_monolithic",))
        with self.assertRaises(AssertionError):
            _pin_enum_search(frag, "role_local_with_scheduler")

    def test_pinned_values_are_defaults(self) -> None:
        # Every pinned value must equal the fragment default (choices[0]) so a
        # default (unset) config is byte-identical whether pinned or not. We only
        # assert the well-known enum defaults here.
        pins = _rank0_pinned_enum_values()
        self.assertEqual(pins["tcgen05_c_stages"], 2)
        self.assertEqual(pins["tcgen05_num_epi_warps"], 4)
        self.assertEqual(pins["tcgen05_strategy"], "role_local_monolithic")
        self.assertEqual(pins["tcgen05_warp_spec_scheduler_warps"], 0)
        self.assertEqual(pins["tcgen05_warp_spec_c_input_warps"], 0)

    def test_config_violates_pin_detects_nondefault_enum(self) -> None:
        self.assertTrue(_config_violates_rank0_pin({"tcgen05_c_stages": 4}))
        self.assertTrue(
            _config_violates_rank0_pin(
                {"tcgen05_strategy": "role_local_with_scheduler"}
            )
        )
        self.assertTrue(
            _config_violates_rank0_pin({"tcgen05_warp_spec_scheduler_warps": 1})
        )

    def test_config_violates_pin_detects_nonpointer_indexing(self) -> None:
        self.assertTrue(
            _config_violates_rank0_pin(
                {"indexing": ["pointer", "tensor_descriptor", "pointer"]}
            )
        )
        self.assertFalse(
            _config_violates_rank0_pin({"indexing": ["pointer", "pointer"]})
        )

    def test_config_does_not_violate_pin_at_defaults(self) -> None:
        # A config that only sets pinned knobs to their pin, or omits them, is not
        # a violation (this is what keeps the pinned SEARCH surface non-empty).
        self.assertFalse(_config_violates_rank0_pin({}))
        self.assertFalse(
            _config_violates_rank0_pin(
                {
                    "tcgen05_c_stages": 2,
                    "tcgen05_strategy": "role_local_monolithic",
                    "tcgen05_warp_spec_scheduler_warps": 0,
                    "tcgen05_warp_spec_c_input_warps": 0,
                    "indexing": ["pointer", "pointer", "pointer"],
                    # TUNE knobs set to non-defaults are NOT violations:
                    "tcgen05_ab_stages": 12,
                    "pid_type": "persistent_interleaved",
                    "l2_groupings": [16],
                }
            )
        )

    def test_pin_listof_indexing(self) -> None:
        # ListOf pinning replaces the inner fragment's search_choices only.
        import dataclasses

        inner = EnumFragment(choices=("pointer", "tensor_descriptor"))
        seq = ListOf(inner=inner, length=5)
        pinned_inner = _pin_enum_search(seq.inner, "pointer")
        pinned_seq = dataclasses.replace(seq, inner=pinned_inner)
        self.assertEqual(pinned_seq.length, 5)
        self.assertEqual(pinned_seq.inner.search_choices, ("pointer",))
        self.assertEqual(pinned_seq.inner.choices, ("pointer", "tensor_descriptor"))


if __name__ == "__main__":
    unittest.main()
