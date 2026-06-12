"""
Python port of domain_specific/manual_policies/blocksworld.cpp

Greedy hand-crafted heuristic for the Blocksworld domain.  Used as a
shaping signal:
  reward += policy_match_reward  if agent_action == expert_action(obs)

Action ordering (matches C++ constants):
  0       : choose_table
  i + 1   : choose_block_i   (0-indexed block i)

Position encoding (per block b):
  0       : in hand
  1       : on the table
  p >= 2  : on another block; the encoding skips b itself:
              x = p - 2         if (p - 2) < b
              x = p - 1         otherwise
  e.g. block 2 on block 0 → position 2; on block 1 → position 3; on block 3 → position 4.

Hand encoding:
  1 : empty
  0 : holding a block

State layout in the Python obs vector:
  obs[0]            : num_blocks  (constant – always first)
  obs[1..5]         : other constants (table_limit, use_hand_empty_flag,
                      use_clear_flags, use_height, use_table_counter)
  obs[6]            : hand        (1 = empty, 0 = holding)
  obs[7]            : table_counter
  obs[8 .. 8+B-1]   : position_of(0) .. position_of(B-1)
  (optional extras beyond 8+B: clear flags, heights – not used by this policy)
"""

ACTION_TABLE = 0  # choose_table / drop on table


def blocksworld_policy(obs) -> int:
    """
    Args:
        obs: flat state vector (list or numpy array of ints/floats).

    Returns:
        Integer action index.
    """
    obs = [int(x) for x in obs]

    N_CONSTANTS = 6
    num_blocks    = obs[0]
    hand          = obs[N_CONSTANTS]          # obs[6]: 1=empty, 0=holding
    table_counter = obs[N_CONSTANTS + 1]      # obs[7]
    pos_start     = N_CONSTANTS + 2           # obs[8]: start of position_of(b)

    hand_empty = (hand == 1)

    def position_of(b: int) -> int:
        return obs[pos_start + b]

    # Decode the block-below given a block index b and its position value p (>=2).
    # The encoding skips b itself:
    #   x = p - 2  if (p - 2) < b
    #   x = p - 1  otherwise
    def block_below_of(b: int, p: int) -> int:
        candidate = p - 2
        return candidate if candidate < b else candidate + 1

    # Held block (-1 if hand is empty)
    held_block = -1
    if not hand_empty:
        for b in range(num_blocks):
            if position_of(b) == 0:
                held_block = b
                break

    def block_on_top_of(b: int) -> int:
        """Return the block sitting directly on top of b, or -1 if clear."""
        for x in range(num_blocks):
            if x == held_block:
                continue
            px = position_of(x)
            if px >= 2 and block_below_of(x, px) == b:
                return x
        return -1

    def top_of_pile(root: int) -> int:
        """Follow the stack upward from root and return the topmost block."""
        top = root
        while True:
            above = block_on_top_of(top)
            if above == -1:
                return top
            top = above

    # ---- Find k: length of the largest correct sub-tower. ----
    # Block 0 correct ↔ on the table (position 1).
    # Block i > 0 correct ↔ on block (i-1), encoded as position i+1.
    k = 0
    while k < num_blocks:
        p = position_of(k)
        correct = (p == 1) if k == 0 else (p == k + 1)
        if not correct:
            break
        k += 1
    if k == num_blocks:
        return ACTION_TABLE  # already solved

    # ---- Find a dump root (a "spare" pile on the table). ----
    dump_root = -1
    for b in range(num_blocks):
        if position_of(b) != 1:
            continue
        if b == 0 and k > 0:
            continue  # correct stack base
        if b == k and block_on_top_of(k) == -1:
            continue  # free slot for block k
        dump_root = b
        break
    dump_top = -1 if dump_root == -1 else top_of_pile(dump_root)

    # ---- Consolidation phase. ----
    # If there are >= 3 table piles AND block 0 isn't placed yet, consolidate
    # everything into a single pile before building the goal tower.
    needs_consolidation = (table_counter >= 3 and position_of(0) != 1)
    if needs_consolidation:
        if hand_empty:
            # Pick up the top of any pile other than the dump pile.
            for b in range(num_blocks):
                if position_of(b) != 1:
                    continue
                if b == dump_root:
                    continue
                return top_of_pile(b) + 1  # choose_block_{top}
        else:
            # Drop on the dump (or on the table to start one).
            return ACTION_TABLE if dump_top == -1 else dump_top + 1

    # ---- Hand holds a block. ----
    if not hand_empty:
        if held_block == k:
            # Place block k in its correct position.
            return ACTION_TABLE if k == 0 else (k - 1) + 1  # on table or on block k-1
        # Wrong block – dump it.
        return ACTION_TABLE if dump_top == -1 else dump_top + 1

    # ---- Hand empty. ----
    # If block k is clear, pick it up; otherwise unstack the dump.
    if block_on_top_of(k) == -1:
        return k + 1  # choose_block_k
    return dump_top + 1  # unstack dump
