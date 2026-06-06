#include "../domain_specific.h"
 
ActionLabel_type InvertedPendulum::get_action(const StateValues& state_values) {
    // TODO: implement InvertedPendulum policy
    // std::cout << "State:"; state_values.dump();
    auto angle = state_values.get_float(2);
    auto angular_velocity = state_values.get_float(3);
    auto value = angle + 0.5 * angular_velocity;
    if (value > 0) return 0; 
    if (value < 0) return 1;
    return 2;
}
