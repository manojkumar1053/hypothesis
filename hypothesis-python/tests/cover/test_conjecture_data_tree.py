# coding=utf-8
#
# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Most of this work is copyright (C) 2013-2019 David R. MacIver
# (david@drmaciver.com), but it contains contributions by others. See
# CONTRIBUTING.rst for a full list of people who may hold copyright, and
# consult the git log if you need to determine who owns an individual
# contribution.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.
#
# END HEADER

from __future__ import absolute_import, division, print_function

from random import Random

from hypothesis import HealthCheck, settings
from hypothesis.internal.compat import hbytes, hrange
from hypothesis.internal.conjecture.data import ConjectureData, Status
from hypothesis.internal.conjecture.engine import ConjectureRunner, RunIsComplete

TEST_SETTINGS = settings(
    max_examples=5000, database=None, suppress_health_check=HealthCheck.all()
)


def runner_for(*examples):
    if len(examples) == 1 and isinstance(examples[0], list):
        examples = examples[0]

    def accept(tf):
        runner = ConjectureRunner(tf, settings=TEST_SETTINGS, random=Random(0))
        ran_examples = []
        for e in examples:
            e = hbytes(e)
            try:
                data = runner.cached_test_function(e)
            except RunIsComplete:
                pass
            ran_examples.append((e, data))
        #       for e, d in ran_examples:
        #           rewritten, status = runner.tree.rewrite(e)
        #           assert status == d.status
        #           assert rewritten == d.buffer
        return runner

    return accept


def test_can_lookup_cached_examples():
    @runner_for(b"\0\0", b"\0\1")
    def runner(data):
        data.draw_bits(8)
        data.draw_bits(8)


def test_can_lookup_cached_examples_with_forced():
    @runner_for(b"\0\0", b"\0\1")
    def runner(data):
        data.write(b"\1")
        data.draw_bits(8)


def test_can_detect_when_tree_is_exhausted():
    @runner_for(b"\0", b"\1")
    def runner(data):
        data.draw_bits(1)

    assert runner.tree.is_exhausted


def test_can_detect_when_tree_is_exhausted_variable_size():
    @runner_for(b"\0", b"\1\0", b"\1\1")
    def runner(data):
        if data.draw_bits(1):
            data.draw_bits(1)

    assert runner.tree.is_exhausted


def test_one_dead_branch():
    @runner_for([[0, i] for i in range(16)] + [[i] for i in range(1, 16)])
    def runner(data):
        i = data.draw_bits(4)
        if i > 0:
            data.mark_invalid()
        data.draw_bits(4)

    assert runner.tree.is_exhausted


def test_non_dead_root():
    @runner_for(b"\0\0", b"\1\0", b"\1\1")
    def runner(data):
        data.draw_bits(1)
        data.draw_bits(1)


def test_can_reexecute_dead_examples():
    @runner_for(b"\0\0", b"\0\1", b"\0\0")
    def runner(data):
        data.draw_bits(1)
        data.draw_bits(1)


def test_novel_prefixes_are_novel():
    def tf(data):
        for _ in range(4):
            data.write(b"\0")
            data.draw_bits(2)

    runner = ConjectureRunner(tf, settings=TEST_SETTINGS, random=Random(0))
    for _ in range(100):
        prefix = runner.tree.generate_novel_prefix(runner.random)
        example = prefix + hbytes(8 - len(prefix))
        assert runner.tree.rewrite(example)[1] is None
        result = runner.cached_test_function(example)
        assert runner.tree.rewrite(example)[0] == result.buffer


def test_overruns_if_not_enough_bytes_for_block():
    runner = ConjectureRunner(
        lambda data: data.draw_bytes(2), settings=TEST_SETTINGS, random=Random(0)
    )
    runner.cached_test_function(b"\0\0")
    assert runner.tree.rewrite(b"\0")[1] == Status.OVERRUN


def test_overruns_if_prefix():
    runner = ConjectureRunner(
        lambda data: [data.draw_bits(1) for _ in range(2)],
        settings=TEST_SETTINGS,
        random=Random(0),
    )
    runner.cached_test_function(b"\0\0")
    assert runner.tree.rewrite(b"\0")[1] == Status.OVERRUN


def test_stores_the_tree_flat_until_needed():
    @runner_for(hbytes(10))
    def runner(data):
        for _ in hrange(10):
            data.draw_bits(1)
        data.mark_interesting()

    root = runner.tree.root
    assert len(root.bits) == 10
    assert len(root.values) == 10
    assert root.transition == Status.INTERESTING


def test_split_in_the_middle():
    @runner_for([0, 0, 2], [0, 1, 3])
    def runner(data):
        data.draw_bits(1)
        data.draw_bits(1)
        data.draw_bits(4)
        data.mark_interesting()

    root = runner.tree.root
    assert len(root.bits) == 2
    assert len(root.values) == 1
    assert list(root.transition[0].values) == [2]
    assert list(root.transition[1].values) == [3]


def test_stores_forced_nodes():
    @runner_for(hbytes(3))
    def runner(data):
        data.draw_bits(1, forced=0)
        data.draw_bits(1)
        data.draw_bits(1, forced=0)
        data.mark_interesting()

    root = runner.tree.root
    assert root.forced == {0, 2}


def test_correctly_relocates_forced_nodes():
    @runner_for([0, 0], [1, 0])
    def runner(data):
        data.draw_bits(1)
        data.draw_bits(1, forced=0)
        data.mark_interesting()

    root = runner.tree.root
    assert root.transition[1].forced == {0}
    assert root.transition[0].forced == {0}
