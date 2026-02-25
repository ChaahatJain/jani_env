#include <unordered_map>
#include "base_components.h"
#include "engine.h"


struct TarjanNode {
    State state;
    int index;
    int lowlink;
    int current_action_id;
    TarjanNode(State s) : state(s), index(-1), lowlink(-1), current_action_id(-1) {}
};


class TarjanOracle {
    JANIEngine* engine;
    // Disable caching if needed to save memory
    bool disable_cache;
    // Reduced memory mode: Empty the cache after every call of stateSafetyWithAction
    bool reduced_memory_mode;
    // Cache the safety results for states
    std::unordered_map<State, std::tuple<bool, int>, StateHasher> cache;
    // The main Tarjan's DFS function
    // Returns a tuple of (is_safe, safe_action_id) 
    // safe_action_id is -1 if no safe actions exist (the state is unsafe), the state is a goal state, or the state is a dead-end
    std::tuple<bool, int> tarjan_dfs(TarjanNode* node, int index,
                    int start_action_id,
                    std::vector<State>& stack,
                    std::unordered_map<State, std::unique_ptr<TarjanNode>, StateHasher>& on_stack_map);
public:
    TarjanOracle(JANIEngine* eng, bool disable_cache = false, bool reduced_memory_mode = false) : engine(eng), disable_cache(disable_cache), reduced_memory_mode(reduced_memory_mode) {
        if (disable_cache) {
            std::cout << "Oracle cache is disabled to save memory." << std::endl;
        }
        if (reduced_memory_mode) {
            std::cout << "Reduced memory mode is enabled: the oracle cache will be cleared after each query." << std::endl;
        }
    }
    
    std::tuple<bool, int> stateSafetyWithAction(const State& state, int start_action_id = -1) {
        /*Check whether a state is safe. Return the safety result and a safe action starting from the state
        start_action_id specifies which action (id) should be visited first. -1 means an arbitrary one*/
        #ifndef NDEBUG
        std::cout << "DEBUG: Checking safety for state: " << state.toString() << std::endl;
        #endif
        // Perform Tarjan's algorithm starting from this state
        std::vector<State> stack;
        if (stack.size() != 0)
            throw std::runtime_error("Stack should be empty at the start of Tarjan's algorithm");
        // Check if a state is on the stack
        std::unordered_map<State, std::unique_ptr<TarjanNode>, StateHasher> on_stack_map;
        std::unique_ptr<TarjanNode> node = std::make_unique<TarjanNode>(state);
        std::tuple<bool, int> result = tarjan_dfs(node.get(), 0, start_action_id, stack, on_stack_map);
        // int safe = std::get<0>(result) ? 1 : 0;
        if (!disable_cache){
            cache[state] = result;
        }
        if (reduced_memory_mode) {
            if (cache.size() > 10000000) { // Clear the cache if it grows too large (this threshold can be adjusted)
                std::unordered_map<State, std::tuple<bool, int>, StateHasher>().swap(cache); // Clear the cache after each query
            }
        }
        #ifndef NDEBUG
        std::cout << "DEBUG: State is marked " << (std::get<0>(result) ? "safe." : "unsafe.") << std::endl;
        #endif
        return result;
    }

    bool isStateActionSafe(const State& state, int action_id) {
        // Check whether a state is safe under a specific action
        std::tuple<bool, int> safety_result = stateSafetyWithAction(state, action_id); // action_id will be the first action to be visited
        // If the action turns out to be a safe one, it will be returned
        return action_id == std::get<1>(safety_result);
    }

    bool isStateSafe(const State& state) {
        // Check whether a state is safe
        std::tuple<bool, int> result = stateSafetyWithAction(state);
        bool safe = std::get<0>(result);
        return safe;
    }

    bool isStateSafeFromVector(const std::vector<double>& state_vector) {
        // Check whether a state (given as a vector) is safe
        State state;
        state = engine->create_state_from_vector(state_vector); // Convert vector to State
        // Perform Tarjan's algorithm
        return isStateSafe(state);
    }

    bool isEngineStateSafe() {
        const State& state = engine->get_current_state();
        return isStateSafe(state);
    }

    std::tuple<bool, int> engineStateSafetyWithAction(int start_action_id = -1) {
        const State& state = engine->get_current_state();
        return stateSafetyWithAction(state, start_action_id);
    }

    bool isEngineStateActionSafe(int action_id) {
        // Check whether the current engine state is safe under a specific action
        const State& state = engine->get_current_state();
        return isStateActionSafe(state, action_id);
    }

    // For testing purposes
    const State getEngineCurrentState() {
        return engine->get_current_state();
    }

    const std::vector<double> getEngineCurrentStateVector() {
        return engine->get_current_state().toRealVector();
    }
};