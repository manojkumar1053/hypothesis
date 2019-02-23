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

"""A module for miscellaneous useful bits and bobs that don't
obviously belong anywhere else. If you spot a better home for
anything that lives here, please move it."""


from __future__ import absolute_import, division, print_function

from hypothesis.internal.compat import hbytes


def replace_all(buffer, replacements):
    """Substitute multiple replacement values into a buffer.

    Replacements is a list of (start, end, value) triples.
    """

    result = bytearray()
    prev = 0
    offset = 0
    for u, v, r in replacements:
        result.extend(buffer[prev:u])
        result.extend(r)
        prev = v
        offset += len(r) - (v - u)
    result.extend(buffer[prev:])
    assert len(result) == len(buffer) + offset
    return hbytes(result)


def pop_random(random, values):
    """Remove a random element of values, possibly changing the ordering of its
    elements."""

    # We pick the element at a random index. Rather than removing that element
    # from the list (which would be an O(n) operation), we swap it to the end
    # and return the last element of the list. This changes the order of
    # the elements, but as long as these elements are only accessed through
    # random sampling that doesn't matter.
    i = random.randrange(0, len(values))
    values[i], values[-1] = values[-1], values[i]
    return values.pop()
