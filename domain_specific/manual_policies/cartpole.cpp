#include "../domain_specific.h"
 
ActionLabel_type Cartpole::get_action(const StateValues& state_values) {
    // TODO: implement Cartpole policy
    auto angle = state_values.get_float(4);
    auto angular_velocity = state_values.get_float(5);
    auto value = angle + 0.5 * angular_velocity;
    if (value > 0) return 1; 
    if (value < 0) return 0;
    return 2;
    (void)state_values;
    return ActionLabel_type{};
}

// ../../build/PlaJA --engine QL_AGENT --model-file ../../../PlaJABenchmarks/benchmarks_archive/icaps26_fault_fixing/jani_models_and_teachers/inverted_pendulum/models/inverted_pendulum.jani --prop 1 --initial-state-enum sample --num-episodes 50 --max-length-episode 2000 --teacher-name InvertedPendulum --additional-properties ../../../PlaJABenchmarks/benchmarks_archive/icaps26_fault_fixing/jani_models_and_teachers/inverted_pendulum/additional_properties/safety_verification/compact_starts_no_predicates/inverted_pendulum/pa_inverted_pendulum_compact_starts_no_predicates.jani --ensemble-interface ../../../PlaJABenchmarks/benchmarks_archive/icaps26_fault_fixing/jani_models_and_teachers/inverted_pendulum/networks/inverted_pendulum/inverted_pendulum_16_16.jani2nnet --print-stats