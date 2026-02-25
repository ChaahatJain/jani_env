#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/filesystem.h>
#include "engine.h"
#include "oracle.h"


namespace nb = nanobind;

NB_MODULE(backend, m) {
    nb::class_<JANIEngine>(m, "JANIEngine")
        .def(nb::init<const std::filesystem::path &,
                      const std::filesystem::path &,
                      const std::filesystem::path &,
                      const std::filesystem::path &,
                      const std::filesystem::path &,
                      int>(),
                    nb::arg("jani_model_path"),
                    nb::arg("jani_property_path"),
                    nb::arg("start_states_path"),
                    nb::arg("objective_path"),
                    nb::arg("failure_property_path"),
                    nb::arg("seed"))
        .def("get_num_actions", &JANIEngine::get_num_actions)
        .def("get_num_variables", &JANIEngine::get_num_variables)
        .def("get_num_constants", &JANIEngine::get_num_constants)
        .def("get_current_action_mask", &JANIEngine::get_current_action_mask)
        .def("get_action_mask_for_obs", &JANIEngine::get_action_mask_for_obs)
        .def("get_constant_vector", &JANIEngine::get_constant_vector)
        .def("get_current_state_vector", &JANIEngine::get_current_state_vector)
        .def("test_guards_for_action", &JANIEngine::testGuardsForAction)
        .def("test_destinations_for_action", &JANIEngine::testDestinationsForAction)
        .def("reach_goal_current", &JANIEngine::reach_goal_current)
        .def("reach_failure_current", &JANIEngine::reach_failure_current)
        .def("get_lower_bounds", &JANIEngine::get_lower_bounds)
        .def("get_upper_bounds", &JANIEngine::get_upper_bounds)
        .def("get_init_state_pool_size", &JANIEngine::get_init_state_pool_size)
        .def("get_goal_condition_size", &JANIEngine::get_goal_condition_size)
        .def("extract_goal_condition", &JANIEngine::extract_goal_condition)
        .def("extract_condition_from_state_vector", &JANIEngine::extract_condition_from_state_vector)
        .def("extract_condition_from_current_state_vector", &JANIEngine::extract_condition_from_current_state_vector)
        .def("reset", &JANIEngine::reset)
        .def("reset_with_index", &JANIEngine::reset_with_index)
        .def("step", &JANIEngine::step);

    nb::class_<TarjanOracle>(m, "TarjanOracle")
        .def(nb::init<JANIEngine*, bool, bool>(), nb::arg("engine"), nb::arg("disable_cache"), nb::arg("reduced_memory_mode"))
        .def("get_engine_current_state_vector", &TarjanOracle::getEngineCurrentStateVector) // For debugging
        .def("is_engine_state_safe", &TarjanOracle::isEngineStateSafe)
        .def("is_engine_state_action_safe", &TarjanOracle::isEngineStateActionSafe)
        .def("is_state_safe_from_vector", &TarjanOracle::isStateSafeFromVector)
        .def("engine_state_safety_with_action", &TarjanOracle::engineStateSafetyWithAction);
}