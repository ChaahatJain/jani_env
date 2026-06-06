#include "../domain_specific.h"
 
struct BelugaActions {
    ActionLabel_type pop_beluga;
    std::vector<ActionLabel_type> push_rack;
    std::vector<ActionLabel_type> pop_rack;
    ActionLabel_type deliver;
    ActionLabel_type send_back;
    ActionLabel_type no_op;

    explicit BelugaActions(int num_racks)
        : pop_beluga(0)
        , deliver(1 + 2 * num_racks)
        , send_back(2 + 2 * num_racks)
        , no_op(3 + 2 * num_racks)
    {
        // push_rack_0, push_rack_1, ..., pop_rack_0, pop_rack_1, ...
        for (int r = 0; r < num_racks; ++r) {
            push_rack.push_back(1 + r);
            pop_rack.push_back(1 + num_racks + r);
        }
    }
};

// State layout helper struct
struct BelugaStateLayout {
    int num_jigs;
    int num_racks;
    int rack_capacity;
    int num_lines;

    int automaton_index;
    int rack_used_start;
    int rack_used_end;
    int rack_slots_start;
    int rack_slots_end;
    int trailer_index;
    int beluga_start;
    int beluga_end;
    int next_line_index;
    int lines_start;
    int lines_end;
    int truck_index;
    int num_swaps_index;

    explicit BelugaStateLayout(const std::unordered_map<ConstantIdType, int>& cv)
        : num_jigs(static_cast<int>(cv.at(1)))
        , num_racks(static_cast<int>(cv.at(2)))
        , rack_capacity(static_cast<int>(cv.at(3)))
        , num_lines(static_cast<int>(cv.at(4)))
        , automaton_index(0)
        , rack_used_start(1)
        , rack_used_end(rack_used_start + num_racks)
        , rack_slots_start(rack_used_end)
        , rack_slots_end(rack_slots_start + num_racks * rack_capacity)
        , trailer_index(rack_slots_end)
        , beluga_start(trailer_index + 1)
        , beluga_end(beluga_start + num_jigs)
        , next_line_index(beluga_end)
        , lines_start(next_line_index + 1)
        , lines_end(lines_start + num_lines)
        , truck_index(lines_end)
        , num_swaps_index(truck_index + 1)
    {}
};

// Helper functions
inline int get_rack_used(const StateValues& sv, const BelugaStateLayout& layout, int rack) {
    return sv.get_int(layout.rack_used_start + rack);
}

inline int get_rack_slot(const StateValues& sv, const BelugaStateLayout& layout, int rack, int slot) {
    return sv.get_int(layout.rack_slots_start + rack * layout.rack_capacity + slot);
}

inline int get_trailer(const StateValues& sv, const BelugaStateLayout& layout) {
    return sv.get_int(layout.trailer_index);
}

inline int get_beluga_jig(const StateValues& sv, const BelugaStateLayout& layout, int jig) {
    return sv.get_int(layout.beluga_start + jig);
}

inline int get_next_line(const StateValues& sv, const BelugaStateLayout& layout) {
    return sv.get_int(layout.next_line_index);
}

inline int get_line(const StateValues& sv, const BelugaStateLayout& layout, int line) {
    return sv.get_int(layout.lines_start + line);
}

inline int get_truck(const StateValues& sv, const BelugaStateLayout& layout) {
    return sv.get_int(layout.truck_index);
}

inline int get_num_swaps(const StateValues& sv, const BelugaStateLayout& layout) {
    return sv.get_int(layout.num_swaps_index);
}

void print_state(const StateValues& sv, const BelugaStateLayout& layout) {
    std::cout << "###### State Values:"; sv.dump();
    std::cout << "=== State Variables ===" << std::endl;
    std::cout << "Automaton: " << sv.get_int(layout.automaton_index) << std::endl;

    std::cout << "Rack used sizes:" << std::endl;
    for (int r = 0; r < layout.num_racks; ++r) {
        std::cout << "  rack[" << r << "] used: " << get_rack_used(sv, layout, r) << std::endl;
    }

    std::cout << "Rack slots:" << std::endl;
    for (int r = 0; r < layout.num_racks; ++r) {
        for (int s = 0; s < layout.rack_capacity; ++s) {
            std::cout << "  rack[" << r << "] slot[" << s << "]: " << get_rack_slot(sv, layout, r, s) << std::endl;
        }
    }

    std::cout << "Trailer: " << get_trailer(sv, layout) << std::endl;

    std::cout << "Beluga jigs:" << std::endl;
    for (int j = 0; j < layout.num_jigs; ++j) {
        std::cout << "  beluga[" << j + 1 << "]: " << get_beluga_jig(sv, layout, j) << std::endl;
    }

    std::cout << "Next line: " << get_next_line(sv, layout) << std::endl;

    std::cout << "Production lines:" << std::endl;
    for (int l = 0; l < layout.num_lines; ++l) {
        std::cout << "  line[" << l << "]: " << get_line(sv, layout, l) << std::endl;
    }

    std::cout << "Truck: " << get_truck(sv, layout) << std::endl;
    std::cout << "Num swaps: " << get_num_swaps(sv, layout) << std::endl;
    std::cout << "======================" << std::endl;
}

ActionLabel_type Beluga::get_action(const StateValues& state_values) {
    const auto& model_info = model_->get_model_information();
    const auto& cv = model_info.get_constants_values();
    const BelugaStateLayout layout(cv);
    const BelugaActions actions(layout.num_racks);

    // Print full state
    print_state(state_values, layout);

    const int next_line   = get_next_line(state_values, layout); // indexing issue
    const int line_val    = get_line(state_values, layout, next_line);
    const int trailer     = get_trailer(state_values, layout);
    const int truck       = get_truck(state_values, layout);

    std::cout << "=== Decision Variables ===" << std::endl;
    std::cout << "next_line: " << next_line << std::endl;
    std::cout << "line_val (line[" << next_line << "]): " << line_val << std::endl;
    std::cout << "trailer: " << trailer << std::endl;
    std::cout << "truck: " << truck << std::endl;
    std::cout << "==========================" << std::endl;

    // Rule 1: next line needs no jig
    if (line_val == 0) {
        std::cout << "Action: no_op (Rule 1: line_val == 0)" << std::endl;
        return actions.no_op;
    }

    // Rule 2: truck has the right jig for next line
    if (truck > 0 && truck == line_val) {
        std::cout << "Action: deliver (Rule 2: truck=" << truck << " == line_val=" << line_val << ")" << std::endl;
        return actions.deliver;
    }

    // Rule 3: truck has wrong jig
    if (truck > 0 && truck != line_val) {
        std::cout << "Action: send_back (Rule 3: truck=" << truck << " != line_val=" << line_val << ")" << std::endl;
        return actions.send_back;
    }

    // Rule 4: trailer has a jig, push to rack with smallest used size
    if (trailer > 0) {
        int best_rack = -1;
        int best_used = std::numeric_limits<int>::max();
        for (int r = 0; r < layout.num_racks; ++r) {
            int used = get_rack_used(state_values, layout, r);
            if (used < layout.rack_capacity && used < best_used) {
                best_used = used;
                best_rack = r;
            }
        }
        if (best_rack >= 0) {
            std::cout << "Action: push_rack[" << best_rack << "] (Rule 4: trailer=" << trailer << ", best_rack=" << best_rack << " used=" << best_used << ")" << std::endl;
            return actions.push_rack[best_rack];
        }
    }

    // Rule 5: needed jig is on a rack and truck is empty
    if (truck == 0) {
        for (int r = 0; r < layout.num_racks; ++r) {
            for (int s = 0; s < layout.rack_capacity; ++s) {
                int slot_val = get_rack_slot(state_values, layout, r, s);
                if (slot_val == line_val) {
                    std::cout << "Action: pop_rack[" << r << "] (Rule 5: found line_val=" << line_val << " at rack[" << r << "] slot[" << s << "])" << std::endl;
                    return actions.pop_rack[r];
                }
            }
        }
    }

    // Rule 6: fallback — pop from beluga
    std::cout << "Action: pop_beluga (Rule 6: fallback)" << std::endl;
    return actions.pop_beluga;
}

// ./PlaJA --engine QL_AGENT --print-stats --evaluation-mode --trace-episodes 50 --save-agent-actions test.json --prop 0 --initial-state-enum sample --num-episodes 50 --max-length-episode 2000 --teacher-name Beluga --model-file ../../PlaJABenchmarks/benchmarks/beluga/models/swap_unsafe/beluga_6_3.jani --ensemble-interface ../../PlaJABenchmarks/benchmarks/beluga/networks/swap_unsafe/beluga_6_3/beluga_6_3_16_16.jani2nnet