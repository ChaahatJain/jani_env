#include "../domain_specific.h"

// Action ordering for Blocksworld:
//   0       : choose_table
//   i + 1   : choose_block_i
//
// Position encoding (per block b):
//   0 : in hand
//   1 : on the table
//   p in [2, B] : on another block. The encoding skips b itself, so:
//     x = p - 2  if (p - 2) < b
//     x = p - 1  otherwise
//   e.g. block 2 sitting on block 0 -> position 2; on block 1 -> position 3;
//        on block 3 -> position 4.
//
// Hand encoding:
//   1 : empty
//   0 : holding a block
 
ActionLabel_type Blocksworld::get_action(const StateValues& state_values) {
    const int num_blocks    = (state_values.get_int_state_size() - 4) / 3;
    const int hand          = state_values.get_int(1);
    const bool hand_empty   = (hand == 1);
    const int table_counter = state_values.get_int(2);
 
    auto position_of = [&](int b) { return state_values.get_int(3 + b); };
 
    // ---- Find k: length of the largest correct sub-tower. ----
    // Block 0 is correct iff it's on the table (position 1).
    // Block i > 0 is correct iff its position decodes to "on block i-1".
    // Using the encoding: block i sitting on block (i-1) has position value
    //   (i-1) + 2 = i + 1   (since i-1 < i, no skip)
    int k = 0;
    while (k < num_blocks) {
        const int p = position_of(k);
        const bool correct = (k == 0) ? (p == 1) : (p == k + 1);
        if (!correct) break;
        ++k;
    }
    if (k == num_blocks) return 0; // already solved
 
    // ---- Identify held block (if any). ----
    int held_block = -1;
    if (!hand_empty) {
        for (int b = 0; b < num_blocks; ++b) {
            if (position_of(b) == 0) { held_block = b; break; }
        }
    }
 
    // Decode the "block below" given a block index b and a position value p (>=2).
    // The encoding skips b itself: positions [2, 3, 4, ...] enumerate the other
    // blocks in index order. So:
    //   x = p - 2  if (p - 2) < b
    //   x = p - 1  if (p - 1) >= b   (equivalently, p - 2 >= b)
    auto block_below_of = [&](int b, int p) {
        const int candidate = p - 2;
        return (candidate < b) ? candidate : candidate + 1;
    };
 
    auto block_on_top_of = [&](int b) {
        for (int x = 0; x < num_blocks; ++x) {
            if (x == held_block) continue;
            const int px = position_of(x);
            if (px >= 2 && block_below_of(x, px) == b) return x;
        }
        return -1;
    };
    auto top_of_pile = [&](int root) {
        int top = root;
        while (true) {
            int above = block_on_top_of(top);
            if (above == -1) return top;
            top = above;
        }
    };
 
    // ---- Find a dump root: any block on the table that isn't block 0
    //      (the correct-stack base) and isn't the free slot holding block k.
    int dump_root = -1;
    for (int b = 0; b < num_blocks; ++b) {
        if (position_of(b) != 1) continue;
        if (b == 0 && k > 0) continue;   // correct stack base
        if (b == k && block_on_top_of(k) == -1) continue; // free slot
        dump_root = b;
        break;
    }
    const int dump_top = (dump_root == -1) ? -1 : top_of_pile(dump_root);
 
    // ---- Consolidation phase (very simple).
    // If there are >= 2 piles on the table AND block 0 isn't yet placed,
    // first consolidate everything into a single pile before anything else.
    const bool needs_consolidation = (table_counter >= 3 && position_of(0) != 1);
    if (needs_consolidation) {
        if (hand_empty) {
            // Pick up the top of any pile other than the chosen dump.
            for (int b = 0; b < num_blocks; ++b) {
                if (position_of(b) != 1) continue;
                if (b == dump_root) continue;
                return top_of_pile(b) + 1;
            }
        } else {
            // Drop on the dump (or on the table to start the dump).
            return (dump_top == -1) ? 0 : dump_top + 1;
        }
    }
 
    // ---- Hand holds a block. ----
    if (!hand_empty) {
        if (held_block == k) {
            return (k == 0) ? 0 : (k - 1) + 1;
        }
        return (dump_top == -1) ? 0 : dump_top + 1;
    }
 
    // ---- Hand empty: block k clear -> pick up; else unstack the dump. ----
    if (block_on_top_of(k) == -1) {
        return k + 1;
    }
    return dump_top + 1;
}