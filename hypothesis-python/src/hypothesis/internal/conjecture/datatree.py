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

import attr

from hypothesis.errors import Flaky, HypothesisException
from hypothesis.internal.compat import hbytes, hrange
from hypothesis.internal.conjecture.data import (
    ConjectureData,
    DataObserver,
    Status,
    StopTest,
)


class PreviouslyUnseenBehaviour(HypothesisException):
    pass


def inconsistent_generation(self):
    raise Flaky(
        "Inconsistent data generation! Data generation behaved differently "
        "between different runs. Is your data generation depending on external "
        "state?"
    )


@attr.s()
class TreeNode(object):
    bits = attr.ib(default=attr.Factory(list))
    values = attr.ib(default=attr.Factory(list))
    transition = attr.ib(default=None)
    forced = attr.ib(default=attr.Factory(set))

    def split_at(self, i):
        """Splits the tree so that it can incorporate
        a decision at the ``draw_bits`` call corresponding
        to position ``i``, or raises ``Flaky`` if that was
        meant to be a forced node."""
        if i in self.forced:
            inconsistent_generation()

        child = TreeNode(
            bits=self.bits[i + 1 :],
            values=self.values[i + 1 :],
            transition=self.transition,
            forced={j - i - 1 for j in self.forced if j > i},
        )
        key = self.values[i]
        del self.values[i:]
        del self.bits[i + 1 :]
        assert len(self.values) == i
        assert len(self.bits) == i + 1
        self.transition = {key: child}
        self.forced = {j for j in self.forced if j < i}


class DataTree(object):
    """Tracks the tree structure of a collection of ConjectureData
    objects, for use in ConjectureRunner."""

    def __init__(self, cap):
        self.cap = cap
        self.root = TreeNode()

    @property
    def is_exhausted(self):
        """Returns True if every possible node is dead and thus the language
        described must have been fully explored."""
        return False

    def generate_novel_prefix(self, random):
        """Generate a short random string that (after rewriting) is not
        a prefix of any buffer previously added to the tree."""
        return hbytes()

    def rewrite(self, buffer):
        """Use previously seen ConjectureData objects to return a tuple of
        the rewritten buffer and the status we would get from running that
        buffer with the test function. If the status cannot be predicted
        from the existing values it will be None."""
        data = ConjectureData.for_buffer(buffer)
        try:
            self.simulate_test_function(data)
            status = data.status
        except PreviouslyUnseenBehaviour:
            status = None
        return (hbytes(data.buffer), status)

    def simulate_test_function(self, data):
        """Run a simulated version of the test function recorded by
        this tree. Note that this does not currently call ``stop_example``
        or ``start_example`` as these are not currently recorded in the
        tree. This will likely change in future."""
        node = self.root
        try:
            while True:
                for i, (n_bits, previous) in enumerate(zip(node.bits, node.values)):
                    v = data.draw_bits(
                        n_bits, forced=node.values[i] if i in node.forced else None
                    )
                    if v != previous:
                        raise PreviouslyUnseenBehaviour()
                if isinstance(node.transition, Status):
                    data.conclude_test(node.transition)
                elif node.transition is None:
                    raise PreviouslyUnseenBehaviour()
                else:
                    assert len(node.bits) == len(node.values) + 1
                    v = data.draw_bits(node.bits[-1])
                    try:
                        node = node.transition[v]
                    except KeyError:
                        raise PreviouslyUnseenBehaviour()
        except StopTest:
            pass

    def new_observer(self):
        return TreeRecordingObserver(self)


def _is_simple_mask(mask):
    """A simple mask is ``(2 ** n - 1)`` for some ``n``, so it has the effect
    of keeping the lowest ``n`` bits and discarding the rest.

    A mask in this form can produce any integer between 0 and the mask itself
    (inclusive), and the total number of these values is ``(mask + 1)``.
    """
    return (mask & (mask + 1)) == 0


class TreeRecordingObserver(DataObserver):
    def __init__(self, tree):
        self.__tree = tree
        self.__current_node = tree.root
        self.__index_in_current_node = 0

    def draw_bits(self, n_bits, forced, value):
        i = self.__index_in_current_node
        self.__index_in_current_node += 1
        node = self.__current_node
        if i < len(node.bits):
            if n_bits != node.bits[i]:
                inconsistent_generation()
        else:
            assert node.transition is None
            node.bits.append(n_bits)
        assert i < len(node.bits)
        if i < len(node.values):
            # Note that we don't check whether a previously
            # forced value is now free. That will be caught
            # if we ever split the node there, but otherwise
            # may pass silently. This is acceptable because it
            # means we skip a hash set lookup on every
            # draw and that's a pretty niche failure mode.
            if forced and i not in node.forced:
                inconsistent_generation()

            if value != node.values[i]:
                node.split_at(i)
                assert i == len(node.values)
                new_node = TreeNode()
                node.transition[value] = new_node
                self.__current_node = new_node
                self.__index_in_current_node = 0
        elif node.transition is None:
            if forced:
                node.forced.add(i)
            node.values.append(value)
        else:
            try:
                self.__current_node = node.transition[value]
            except KeyError:
                self.__current_node = node.transition.setdefault(value, TreeNode())
            except TypeError:
                assert (
                    isinstance(node.transition, Status)
                    and node.transition != Status.OVERRUN
                )
                inconsistent_generation()
            self.__index_in_current_node = 0

    def conclude_test(self, status, interesting_origin):
        """Says that ``status`` occurred at node ``node``. This updates the
        node if necessary and checks for consistency."""
        if status == Status.OVERRUN:
            return
        i = self.__index_in_current_node
        node = self.__current_node

        if i < len(node.values) or isinstance(node.transition, dict):
            inconsistent_generation()

        if node.transition is not None:
            if node.transition != status:
                raise Flaky(
                    "Inconsistent test results! Test case was %s on first run but %s on second"
                    % (existing.status.name, status)
                )
        else:
            node.transition = status
