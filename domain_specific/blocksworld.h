//
// This file is part of the PlaJA code base.
// Copyright (c) (2019 - 2022) Marcel Vinzent.
// See README.md in the top-level directory for licensing information.
//

#ifndef PLAJA_BLOCKSWORLD_H
#define PLAJA_BLOCKSWORLD_H

#ifndef NDEBUG // only for debugging purposes !!!

#include "../search/states/state_values.h"

bool check_valid(StateValues& state_values) {
    std::vector<std::vector<bool>> is_above(6, std::vector<bool>(6, false));

    for (int i = 0; i < 6; ++i) {
        if (state_values[i + 3] < 2) continue; // not on block
        int block_index;
        if (state_values[i + 3] < i + 2) {
            block_index = state_values[i + 3] - 2;
        } else {
            block_index = state_values[i + 3] - 1;
        }
        is_above[i][block_index] = true;
    }

    // propagate:
    bool found_new = true;
    while (found_new) {
        found_new = false;
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                if (is_above[i][j]) {
                    for (int k = 0; k < 6; ++k) {
                        if (is_above[j][k]) {
                            if (!is_above[i][k]) {
                                found_new = true;
                                is_above[i][k] = true;
                            }
                        }
                    }
                }
            }}}

    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            if (is_above[i][j] && is_above[j][i]) return false;
        }}

    return true;
}

#endif

#endif //PLAJA_BLOCKSWORLD_H
