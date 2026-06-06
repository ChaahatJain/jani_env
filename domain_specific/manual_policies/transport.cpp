#include "../domain_specific.h"

// Action ordering for Transport:
//   0: pick up
//   1: drop
//   2: move forward
//   3: move backward
namespace {
    constexpr int ACTION_PICKUP   = 0;
    constexpr int ACTION_DROP     = 1;
    constexpr int ACTION_FORWARD  = 2;
    constexpr int ACTION_BACKWARD = 3;
}


ActionLabel_type Transport::get_action(const StateValues& state_values) {
    // State layout (offset by 1 for the automaton location):
    //   index 0                       : automaton state
    //   index 1 .. number_of_locations: package count at each location
    //   index number_of_locations + 1 : truck position
    //   index number_of_locations + 2 : truck load
    std::cout << "State:"; state_values.dump();
    const int number_of_locations   = state_values.get_int_state_size() - 4;
    const int truck_position        = state_values.get_int(number_of_locations + 1);
    const int truck_load            = state_values.get_int(number_of_locations + 2);
    const int goal_position         = number_of_locations - 1;
    const int bridge_start_position = number_of_locations - 2;
 
    std::cout << "Num locations: " << number_of_locations << " truck position: " << truck_position << " truck load: " << truck_load << std::endl;
    // Package count at the truck's current location.
    // Locations are stored at indices 1 .. number_of_locations, so location k
    // lives at index k + 1.
    const int packages_here = state_values.get_int(truck_position + 1);
 
    // ---- Rule 1: drop excess load at the bridge start. ----
    // If we're at the bridge start carrying more than one package, drop one.
    // To be safe, We only want to cross the bridge with a single package.
    if (truck_load > 1 && truck_position == bridge_start_position) {
        return ACTION_DROP;
    }
 
    // ---- Rule 2: deliver at the goal. ----
    if (truck_load >= 1 && truck_position == goal_position) {
        return ACTION_DROP;
    }
 
    // ---- Rule 3: carrying something, not at goal -> drive forward. ----
    if (truck_load >= 1) {
        return ACTION_FORWARD;
    }
 
    // ---- Rule 4: empty truck, package at current cell -> pick it up. ----
    if (packages_here > 0 && truck_position != goal_position) {
        return ACTION_PICKUP;
    }
 
    // ---- Rule 5: empty truck, nothing here.
    // If there's a package behind us, go back to get it; otherwise go forward.
    bool package_behind = false;
    for (int loc = 0; loc < truck_position; ++loc) {
        if (state_values.get_int(loc + 1) > 0) {
            package_behind = true;
            break;
        }
    }
    return package_behind ? ACTION_BACKWARD : ACTION_FORWARD;
}
