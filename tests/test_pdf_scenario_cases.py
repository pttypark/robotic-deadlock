"""Tests for the numbered PDF scenario cases."""

import os
import sys
from collections import Counter

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)

from rware.fcfs_cross_simulation import FCFSCrossExperiment
from scripts.run_total_time_bo_policy_comparison import (
    DIRECTIONS,
    DOMINANT_COUNT_RANGE_BY_TOTAL_AGVS,
    EARLY_DEPARTURE_RANGE,
    LATE_DEPARTURE_RANGE,
    NORMAL_DEPARTURE_STEP,
    PDF_CASE_TOTAL_AGV_OPTIONS,
    PDF_TRAIN_TOTAL_AGVS,
    _generate_pdf_case_scenarios,
)


def test_pdf_train_uses_cases_1_to_4_only_with_20_agvs():
    scenarios = _generate_pdf_case_scenarios(
        case_ids=(1, 2, 3, 4),
        scenarios_per_case=2,
        total_agv_options=(PDF_TRAIN_TOTAL_AGVS,),
        seed_base=100,
        scenario_seed=200,
        split="train",
    )

    assert Counter(scenario.scenario_case_id for scenario in scenarios) == {
        1: 2,
        2: 2,
        3: 2,
        4: 2,
    }
    assert all(scenario.total_agvs == PDF_TRAIN_TOTAL_AGVS for scenario in scenarios)
    assert all(scenario.scenario_split == "train_test" for scenario in scenarios)


def test_pdf_test_includes_cases_1_to_5_for_16_20_and_24_agvs():
    scenarios = _generate_pdf_case_scenarios(
        case_ids=(1, 2, 3, 4, 5),
        scenarios_per_case=1,
        total_agv_options=PDF_CASE_TOTAL_AGV_OPTIONS,
        seed_base=100,
        scenario_seed=200,
        split="test",
    )

    assert len(scenarios) == 15
    assert Counter(scenario.total_agvs for scenario in scenarios) == {
        16: 5,
        20: 5,
        24: 5,
    }
    assert [scenario.scenario_case_id for scenario in scenarios[:5]] == [1, 2, 3, 4, 5]
    assert [scenario.scenario_case_id for scenario in scenarios[5:10]] == [1, 2, 3, 4, 5]
    assert [scenario.scenario_case_id for scenario in scenarios[10:]] == [1, 2, 3, 4, 5]
    assert all(
        scenario.scenario_split == "test_only"
        for scenario in scenarios
        if scenario.scenario_case_id == 5
    )


def test_pdf_cases_preserve_total_16_20_and_24_agvs():
    scenarios = _generate_pdf_case_scenarios(
        case_ids=(1, 2, 3, 4, 5),
        scenarios_per_case=2,
        total_agv_options=PDF_CASE_TOTAL_AGV_OPTIONS,
        seed_base=100,
        scenario_seed=200,
        split="test",
    )

    for scenario in scenarios:
        assert sum(scenario.allocation.values()) == scenario.total_agvs
        assert sum(_exit_counts(scenario.goal_plan).values()) == scenario.total_agvs


def test_scenario_1_normal_balanced_has_balanced_counts_and_4_step_departures():
    scenario = _generate_pdf_case_scenarios(
        case_ids=(1,),
        scenarios_per_case=1,
        total_agv_options=(PDF_TRAIN_TOTAL_AGVS,),
        seed_base=100,
        scenario_seed=200,
        split="train",
    )[0]
    balanced_count = scenario.total_agvs // len(DIRECTIONS)

    assert set(scenario.allocation.values()) == {balanced_count}
    assert _exit_counts(scenario.goal_plan) == {
        direction: balanced_count
        for direction in DIRECTIONS
    }
    assert all(
        starts == [index * NORMAL_DEPARTURE_STEP for index in range(balanced_count)]
        for starts in scenario.spawn_plan.values()
    )


def test_scenario_2_arrival_burst_has_half_early_departures():
    scenario = _generate_pdf_case_scenarios(
        case_ids=(2,),
        scenarios_per_case=1,
        total_agv_options=(PDF_TRAIN_TOTAL_AGVS,),
        seed_base=100,
        scenario_seed=200,
        split="train",
    )[0]
    starts = [step for steps in scenario.spawn_plan.values() for step in steps]
    early = [
        step
        for step in starts
        if EARLY_DEPARTURE_RANGE[0] <= step <= EARLY_DEPARTURE_RANGE[1]
    ]
    late = [
        step
        for step in starts
        if LATE_DEPARTURE_RANGE[0] <= step <= LATE_DEPARTURE_RANGE[1]
    ]
    balanced_count = scenario.total_agvs // len(DIRECTIONS)

    assert scenario.early_departures == scenario.total_agvs // 2
    assert len(early) == scenario.total_agvs // 2
    assert len(late) == scenario.total_agvs // 2
    assert set(scenario.allocation.values()) == {balanced_count}
    assert _exit_counts(scenario.goal_plan) == {
        direction: balanced_count
        for direction in DIRECTIONS
    }


def test_scenario_3_direction_skew_rotates_and_samples_dominant_approach_count():
    for total_agvs in PDF_CASE_TOTAL_AGV_OPTIONS:
        scenarios = _generate_pdf_case_scenarios(
            case_ids=(3,),
            scenarios_per_case=4,
            total_agv_options=(total_agvs,),
            seed_base=100,
            scenario_seed=200,
            split="test",
        )
        lower, upper = DOMINANT_COUNT_RANGE_BY_TOTAL_AGVS[total_agvs]
        balanced_exit_count = total_agvs // len(DIRECTIONS)

        assert [scenario.dominant_approach for scenario in scenarios] == [
            "NORTH",
            "EAST",
            "SOUTH",
            "WEST",
        ]
        for scenario in scenarios:
            dominant_count = scenario.allocation[scenario.dominant_approach]
            other_counts = [
                count
                for direction, count in scenario.allocation.items()
                if direction != scenario.dominant_approach
            ]

            assert lower <= dominant_count <= upper
            assert scenario.dominant_approach_count == dominant_count
            _assert_other_counts_bounded(other_counts, total_agvs - dominant_count)
            assert _exit_counts(scenario.goal_plan) == {
                direction: balanced_exit_count
                for direction in DIRECTIONS
            }


def test_scenario_4_exit_concentrated_rotates_and_samples_dominant_exit_count():
    for total_agvs in PDF_CASE_TOTAL_AGV_OPTIONS:
        scenarios = _generate_pdf_case_scenarios(
            case_ids=(4,),
            scenarios_per_case=4,
            total_agv_options=(total_agvs,),
            seed_base=100,
            scenario_seed=200,
            split="test",
        )
        lower, upper = DOMINANT_COUNT_RANGE_BY_TOTAL_AGVS[total_agvs]
        balanced_approach_count = total_agvs // len(DIRECTIONS)

        assert [scenario.dominant_exit for scenario in scenarios] == [
            "NORTH",
            "EAST",
            "SOUTH",
            "WEST",
        ]
        for scenario in scenarios:
            exit_counts = _exit_counts(scenario.goal_plan)
            dominant_count = exit_counts[scenario.dominant_exit]
            other_counts = [
                count
                for direction, count in exit_counts.items()
                if direction != scenario.dominant_exit
            ]

            assert set(scenario.allocation.values()) == {balanced_approach_count}
            assert lower <= dominant_count <= upper
            assert scenario.dominant_exit_count == dominant_count
            _assert_other_counts_bounded(other_counts, total_agvs - dominant_count)


def test_scenario_5_mixed_combines_burst_approach_skew_and_exit_concentration():
    for total_agvs in PDF_CASE_TOTAL_AGV_OPTIONS:
        scenario = _generate_pdf_case_scenarios(
            case_ids=(5,),
            scenarios_per_case=1,
            total_agv_options=(total_agvs,),
            seed_base=100,
            scenario_seed=200,
            split="test",
        )[0]
        starts = [step for steps in scenario.spawn_plan.values() for step in steps]
        early = [
            step
            for step in starts
            if EARLY_DEPARTURE_RANGE[0] <= step <= EARLY_DEPARTURE_RANGE[1]
        ]
        exit_counts = _exit_counts(scenario.goal_plan)
        lower, upper = DOMINANT_COUNT_RANGE_BY_TOTAL_AGVS[total_agvs]

        assert scenario.scenario_split == "test_only"
        assert scenario.dominant_approach == "NORTH"
        assert scenario.dominant_exit == "EAST"
        assert lower <= scenario.dominant_approach_count <= upper
        assert lower <= scenario.dominant_exit_count <= upper
        assert scenario.allocation[scenario.dominant_approach] == scenario.dominant_approach_count
        assert exit_counts[scenario.dominant_exit] == scenario.dominant_exit_count
        _assert_other_counts_bounded(
            [
                count
                for direction, count in scenario.allocation.items()
                if direction != scenario.dominant_approach
            ],
            total_agvs - scenario.dominant_approach_count,
        )
        _assert_other_counts_bounded(
            [
                count
                for direction, count in exit_counts.items()
                if direction != scenario.dominant_exit
            ],
            total_agvs - scenario.dominant_exit_count,
        )
        assert len(early) == total_agvs // 2


def _assert_other_counts_bounded(counts, remaining_total):
    assert len(counts) == 3
    assert all(count >= 1 for count in counts)
    assert sum(counts) == remaining_total
    assert all(count <= remaining_total // 2 for count in counts)


def _exit_counts(goal_plan):
    by_node = Counter(goal for goals in goal_plan.values() for goal in goals)
    return {
        direction: by_node[node]
        for node, direction in FCFSCrossExperiment.EXIT_DIRECTION_BY_NODE.items()
    }
