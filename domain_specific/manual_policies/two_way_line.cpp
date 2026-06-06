#include "../domain_specific.h"

/* Note: This manual policy is not necessarily safe for the one way line Icy domain. This is because packages can be nondeterministically dropped during moving. This is highly unlikely and considering this in the policy makes it reach the goal very slowly.*/

// Action ordering for TwoWayLine:
namespace {
    constexpr int ACTION_PICKUP   = 0;
    constexpr int ACTION_DROP     = 1;
    constexpr int ACTION_ACCELERATE  = 2;
    constexpr int ACTION_DECELERATE = 3;
    constexpr int ACTION_MOVE = 4;
}


ActionLabel_type TwoWayLine::get_action(const StateValues& state_values) {
    // State layout (offset by 1 for the automaton location):
    //   index 0                       : automaton state
    //   index 1 .. number_of_locations: package count at each location
    //   index number_of_locations + 1 : truck position
    //   index number_of_locations + 2 : truck load
    //   index number_of_locations + 3: truck_velocity
    std::cout << "State:"; state_values.dump();
    const int number_of_locations   = state_values.get_int_state_size() - 5;
    const int truck_position        = state_values.get_int(number_of_locations + 1);
    const int truck_load            = state_values.get_int(number_of_locations + 2);
    const int truck_velocity        = state_values.get_int(number_of_locations + 3);
    const int goal_position         = number_of_locations - 1;
 
    std::cout << "Num locations: " << number_of_locations << " truck position: " << truck_position << " truck load: " << truck_load << std::endl;
    // Package count at the truck's current location.
    // Locations are stored at indices 1 .. number_of_locations, so location k
    // lives at index k + 1.
    const int packages_here = state_values.get_int(truck_position + 1);

    const int packages_next =
        (truck_position + 1 < number_of_locations)
            ? state_values.get_int((truck_position + 1) + 1)
            : 0;
    
    const int packages_prev = 
        (truck_position >= 0)
            ? state_values.get_int(truck_position + 1)
            : 0;

    bool package_behind = false;
    for (int loc = 0; loc < truck_position; ++loc) {
        if (state_values.get_int(loc + 1) > 0) {
            package_behind = true;
            break;
        }
    }
 
    // ---- Rule: at goal with velocity 0 -> drop. ----
    if (truck_position == goal_position && truck_velocity == 0 && truck_load > 0) {
        return ACTION_DROP;
    }
 
    // ---- Rule: at goal - 1 with velocity 1 -> decelerate (so we arrive stopped). ----
    if (truck_position == goal_position - 1 && truck_velocity == 1) {
        return ACTION_DECELERATE;
    }
 
    // ---- Rule: moving with a package on the next cell -> decelerate to stop and pick up. ----
    if (truck_velocity == 1 && packages_next > 0) {
        return ACTION_DECELERATE;
    }

    if (truck_velocity == -1 && packages_prev > 0) {
        return ACTION_ACCELERATE;
    }

    // Gap-fill: stopped on top of a package -> pick it up.
    if (truck_velocity == 0 && packages_here > 0 && truck_position != goal_position) {
        return ACTION_PICKUP;
    }

    // --- Rule: Go back for missing packages.
    if (truck_velocity >= 0 && package_behind) {
        return ACTION_DECELERATE;
    }
 
    // ---- Rule: moving with nothing to stop for -> keep coasting. ----
    if (truck_velocity == 1 or truck_velocity == -1) {
        return ACTION_MOVE;
    }

    if (truck_velocity > 1) {
        return ACTION_DECELERATE;
    }

    if (truck_velocity < -1) {
        return ACTION_ACCELERATE;
    }
 
    // ---- Velocity 0 from here on. ----
 
    
 
    // ---- Rule: velocity 0, no package here -> accelerate.
    // (Also covers velocity 0 while carrying a package: get moving toward goal.)
    return ACTION_ACCELERATE;
}

// ./PlaJA --engine QL_AGENT --print-stats --evaluation-mode --trace-episodes 50 --save-agent-actions test.json --prop 1 --initial-state-enum sample --num-episodes 3 --max-length-episode 2000 --teacher-name TwoWayLine --model-file ../../PlaJABenchmarks/benchmarks_archive/icaps26_fault_fixing/jani_models_and_teachers/two_way_line/models/non_det_no_park/two_way_line.jani --additional-properties ../../PlaJABenchmarks/benchmarks_archive/icaps26_fault_fixing/jani_models_and_teachers/two_way_line/additional_properties/safety_verification/non_det_no_park/compact_starts_no_predicates/two_way_line/pa_two_way_line_compact_starts_no_predicates.jani --ensemble-interface ../../PlaJABenchmarks/benchmarks_archive/icaps26_fault_fixing/jani_models_and_teachers/two_way_line/networks/non_det_no_park/two_way_line/two_way_line_64_64.jani2nnet 