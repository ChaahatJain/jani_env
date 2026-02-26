#include <unordered_map>
#include <fstream>
#include <random>
#include <assert.h>
#include "oracle.h"


size_t rss_mb() {
    std::ifstream f("/proc/self/statm");
    size_t size, resident;
    f >> size >> resident;
    return (resident * 4) / 1024; // MB (page size = 4KB usually)
}


void print_indent(int indent) {
    for (int i = 0; i < indent; i++)
        std::cout << "  ";
}

std::vector<int> random_permutation(int n) {
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(perm.begin(), perm.end(), g);
    return perm;
}

void pop_up_stack_until(
    std::vector<State>& stack,
    std::unordered_map<State, std::unique_ptr<TarjanNode>, StateHasher>& on_stack_map,
    const State& target_state,
    bool inclusive) {
        while (true) {
            if (stack.empty()) {
                throw std::runtime_error("Stack is empty before reaching target state");
            }
            State top_state = stack.back();
            if (top_state == target_state) {
                if (inclusive) {
                    stack.pop_back();
                    on_stack_map.erase(top_state);
                }
                break;
            }
            stack.pop_back();
            on_stack_map.erase(top_state);
        }
    }


std::tuple<bool, int>TarjanOracle::tarjan_dfs(
    TarjanNode *node, int index,
    int start_action_id,
    std::vector<State> &stack, 
    std::unordered_map<State, std::unique_ptr<TarjanNode>, StateHasher> &on_stack_map) {
        if ((node->index != -1) || (node->lowlink != -1)) 
        throw std::runtime_error("Node should not be initialized before");

        #ifndef NDEBUG
        print_indent(index);
        std::cout << "DEBUG: Visiting state: " << node->state.toString() << std::endl;
        #endif

        node->index = stack.size();
        node->lowlink = stack.size();

        #ifndef NDEBUG
        print_indent(index);
        std::cout << "DEBUG: Assigned index and lowlink: " << index << std::endl;
        #endif  

        if (cache.find(node->state) != cache.end()) { 

            #ifndef NDEBUG
            print_indent(index);
            std::cout << "DEBUG: State found in cache." << std::endl;
            print_indent(index);
            std::cout << "DEBUG: State marked " << (std::get<0>(cache[node->state]) ? "safe." : "unsafe.") << " with action " << std::get<1>(cache[node->state]) << std::endl;
            print_indent(index);
            std::cout << "DEBUG: Exiting state: " << node->state.toString() << std::endl;
            #endif
            
            if (start_action_id == -1)
                return cache[node->state]; // If the state has been cached
        }
        // TODO: Check whether this inputs the reference of the state
        if (engine->reach_goal(node->state)) {
            #ifndef NDEBUG
            print_indent(index);
            std::cout << "DEBUG: State is a goal state, marked safe." << std::endl;
            print_indent(index);
            std::cout << "DEBUG: Exiting state: " << node->state.toString() << std::endl;
            #endif
            return std::make_tuple(true, -1); // A goal state is a safe state
        }

        if (engine->reach_failure(node->state)) {
            #ifndef NDEBUG
            print_indent(index);
            std::cout << "DEBUG: State is a failure state, marked unsafe." << std::endl;
            print_indent(index);
            std::cout << "DEBUG: Exiting state: " << node->state.toString() << std::endl;
            #endif
            return std::make_tuple(false, -1); // A failure state is an unsafe state
        }

        // Push the state onto the stack
        stack.push_back(node->state);
        // Create a copy of the node for storing in the on-stack map
        on_stack_map[node->state] = std::make_unique<TarjanNode>(node->state);
        on_stack_map[node->state]->index = node->index;
        on_stack_map[node->state]->lowlink = node->lowlink;
        
        // Iterate through all applicable actions
        int num_actions = engine->get_num_actions();

        // Randomly permute the action ids
        // std::vector<int> permuted_action_ids = random_permutation(num_actions);
        std::vector<int> permuted_action_ids(num_actions); // Let's try keeping the original order
        std::iota(permuted_action_ids.begin(), permuted_action_ids.end(), 0);
        std::vector<int> action_visit_seq;
        if (start_action_id != -1)
            action_visit_seq.push_back(start_action_id);
        for (int action_id : permuted_action_ids) {
            if (action_id == start_action_id)
                continue;
            action_visit_seq.push_back(action_id);
        }
        if (action_visit_seq.size() != num_actions) {
            throw std::runtime_error("Number " + std::to_string(action_visit_seq.size()) + " of actions to be visited is different from the total number of actions " + std::to_string(num_actions));
        }
        if (action_visit_seq.size() != permuted_action_ids.size()) {
            throw std::runtime_error("Number " + std::to_string(action_visit_seq.size()) + " of actions to be visited is different from the actual number of actions " + std::to_string(permuted_action_ids.size()));
        }
        if ((start_action_id != -1) && (action_visit_seq[0] != start_action_id))
            throw std::runtime_error("First action to be visited is different from the scheduled one");

        std::vector<bool> action_mask = engine->get_action_mask(node->state);
        bool is_safe_state = true; // If no action is applicable, the state is safe
        int safe_action_id = -1; // To record one safe action id
        for (int action_id : action_visit_seq) {

            #ifndef NDEBUG
            print_indent(index);
            std::cout << "DEBUG: Checking action id " << action_id;
            #endif

            if (!action_mask[action_id]) {
                #ifndef NDEBUG
                std::cout << " -- Action not applicable" << std::endl;
                #endif
                continue; // Continue to the next action if it is not applicable
            }
            #ifndef NDEBUG
            std::cout << " -- Action applicable" << std::endl;
            #endif

            // Ensure all successors are poped from the stack before processing the next action (current state must remain)
            pop_up_stack_until(stack, on_stack_map, node->state, false);
            if (!(stack.back() == node->state))
                throw std::runtime_error("Top of stack does not match current node state before processing action");

            bool is_safe_action = true;
            if (on_stack_map.find(node->state) == on_stack_map.end())
                throw std::runtime_error("Current node not found in on-stack map");
            on_stack_map[node->state]->current_action_id = action_id; // update current action id

            std::vector<State> successor_states = engine->get_all_successor_states(node->state, action_id);

            #ifndef NDEBUG
            print_indent(index);
            std::cout << "DEBUG: Number of successor states: " << successor_states.size() << std::endl;
            int succ_idx = 0;
            #endif

            for (State& succ_state : successor_states) {
                if (on_stack_map.find(succ_state) != on_stack_map.end()) {

                    #ifndef NDEBUG
                    print_indent(index);
                    std::cout << "DEBUG: Successor state " << succ_idx << " is on stack: " << succ_state.toString() << std::endl;
                    succ_idx++;
                    #endif

                    // Successor is on stack, update lowlink
                    TarjanNode* succ_node = on_stack_map[succ_state].get();
                    node->lowlink = std::min(node->lowlink, succ_node->index);

                    #ifndef NDEBUG
                    print_indent(index);
                    std::cout << "DEBUG: Updated lowlink to " << node->lowlink << std::endl;
                    #endif

                    // Update the copy of on-stack map
                    if (on_stack_map.find(node->state) == on_stack_map.end())
                        throw std::runtime_error("Current node not found in on-stack map");
                    on_stack_map[node->state]->lowlink = node->lowlink;

                    #ifndef NDEBUG
                    print_indent(index);
                    std::cout << "DEBUG: Current node lowlink in on-stack map updated to " << on_stack_map[node->state]->lowlink << std::endl;
                    #endif

                } else {

                    #ifndef NDEBUG
                    print_indent(index);
                    std::cout << "DEBUG: Successor state " << succ_idx << " is not on stack: " << succ_state.toString() << std::endl;
                    #endif

                    // Successor not on stack
                    std::unique_ptr<TarjanNode> succ_node = std::make_unique<TarjanNode>(succ_state);
                    // TarjanNode* succ_node = new TarjanNode(succ_state);
                    // Action visiting order is irrevelent after the first call of the dfs
                    std::tuple<bool, int> succ_result = tarjan_dfs(succ_node.get(), index + 1, -1, stack, on_stack_map);
                    bool succ_safe = std::get<0>(succ_result);

                    #ifndef NDEBUG
                    print_indent(index);
                    std::cout << "DEBUG: Successor state " << succ_node->state.toString() << " returned " << (succ_safe ? "safe." : "unsafe.") << " with action " << std::get<1>(succ_result) << std::endl;
                    succ_idx++;
                    #endif

                    node->lowlink = std::min(node->lowlink, succ_node->lowlink);

                    #ifndef NDEBUG
                    print_indent(index);
                    std::cout << "DEBUG: Updated lowlink to " << node->lowlink << std::endl;
                    #endif

                    // Again, update the copy of on-stack map
                    if (on_stack_map.find(node->state) == on_stack_map.end())
                        throw std::runtime_error("Current node not found in on-stack map");
                    on_stack_map[node->state]->lowlink = node->lowlink;

                    #ifndef NDEBUG
                    print_indent(index);
                    std::cout << "DEBUG: Current node lowlink in on-stack map updated to " << on_stack_map[node->state]->lowlink << std::endl;
                    #endif 

                    if (!succ_safe) {
                        is_safe_action = false;
                        break; // No need to check other successors for this action
                    }
                }
            }
            if (on_stack_map[node->state]->current_action_id != action_id)
                throw std::runtime_error("Current action id in on-stack map does not match");

            if (is_safe_action) {
                safe_action_id = action_id;
                if (node->lowlink == node->index) {
                    // If the current node is the root of an SCC
                    if (!disable_cache){
                        // Mark the state as safe in the cache only for the root of the SCC
                        cache[node->state] = std::make_tuple(true, action_id);
                    }
                    // Pop the stack up to and including the current node
                    pop_up_stack_until(stack, on_stack_map, node->state, true);
                }

                #ifndef NDEBUG
                print_indent(index);
                std::cout << "DEBUG: Exiting state: " << node->state.toString() << " marked safe with action " << action_id << std::endl;
                #endif

                return std::make_tuple(true, safe_action_id);
            }
        }

        if (!disable_cache) {
            // If we reach here, then the state is unsafe
            cache[node->state] = std::make_tuple(false, -1); // Directly cache unsafe state
        }
        
        if (node->lowlink == node->index)
            pop_up_stack_until(stack, on_stack_map, node->state, true);

        #ifndef NDEBUG
            print_indent(index);
            std::cout << "DEBUG: Exiting state: " << node->state.toString() << " marked unsafe" << std::endl;
        #endif
        
        return std::make_tuple(false, -1);
    }