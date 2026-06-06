#include "../domain_specific.h"
 
ActionLabel_type BouncingBall::get_action(const StateValues& state_values) {
    // TODO: implement BouncingBall policy
    // std::cout << "State"; state_values.dump(nullptr);
    return 0; // Always select to push the ball if possible.
}
