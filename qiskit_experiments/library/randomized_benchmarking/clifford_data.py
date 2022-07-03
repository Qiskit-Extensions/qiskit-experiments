# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
This file contains the Clifford group represented as integers.
In CLIFF_COMPOSE_DATA, (i, j): k represents Clifford(i).compose(clifford(j)) = Clifford(k).
Since retrieving a value from an array is more efficient than from a dict, therefore
we store only the results in an array. The index is computed in c
CliffordUtils.compose_num_with_clifford().
Note that for the pairs (i, j), i can be any clifford, j represents only the
1-gate cliffords, as listed in CliffordUtils.general_cliff_list
"""

CLIFF_COMPOSE_DATA = [
    0,
    1,
    2,
    4,
    6,
    8,
    12,
    18,
    22,
    1,
    0,
    3,
    5,
    7,
    9,
    13,
    19,
    23,
    2,
    23,
    6,
    15,
    8,
    0,
    14,
    20,
    9,
    3,
    22,
    7,
    14,
    9,
    1,
    15,
    21,
    8,
    4,
    9,
    17,
    18,
    10,
    23,
    16,
    22,
    0,
    5,
    8,
    16,
    19,
    11,
    22,
    17,
    23,
    1,
    6,
    19,
    8,
    16,
    0,
    2,
    18,
    12,
    10,
    7,
    18,
    9,
    17,
    1,
    3,
    19,
    13,
    11,
    8,
    5,
    0,
    3,
    2,
    6,
    20,
    14,
    21,
    9,
    4,
    1,
    2,
    3,
    7,
    21,
    15,
    20,
    10,
    15,
    23,
    6,
    4,
    17,
    22,
    16,
    12,
    11,
    14,
    22,
    7,
    5,
    16,
    23,
    17,
    13,
    12,
    13,
    20,
    10,
    18,
    14,
    0,
    6,
    16,
    13,
    12,
    21,
    11,
    19,
    15,
    1,
    7,
    17,
    14,
    11,
    12,
    21,
    20,
    18,
    2,
    8,
    3,
    15,
    10,
    13,
    20,
    21,
    19,
    3,
    9,
    2,
    16,
    21,
    11,
    12,
    22,
    5,
    4,
    10,
    6,
    17,
    20,
    10,
    13,
    23,
    4,
    5,
    11,
    7,
    18,
    7,
    14,
    22,
    12,
    20,
    6,
    0,
    4,
    19,
    6,
    15,
    23,
    13,
    21,
    7,
    1,
    5,
    20,
    17,
    18,
    9,
    14,
    12,
    8,
    2,
    15,
    21,
    16,
    19,
    8,
    15,
    13,
    9,
    3,
    14,
    22,
    3,
    5,
    0,
    16,
    11,
    10,
    4,
    18,
    23,
    2,
    4,
    1,
    17,
    10,
    11,
    5,
    19,
]

# In CLIFF_INVERSE_DATA, i: j represents Clifford(i).inverse = Clifford(j)
CLIFF_INVERSE_DATA = [
    0,
    1,
    8,
    5,
    22,
    3,
    6,
    19,
    2,
    23,
    10,
    15,
    12,
    13,
    14,
    11,
    16,
    21,
    18,
    7,
    20,
    17,
    4,
    9,
]
