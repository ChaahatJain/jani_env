#include <gtest/gtest.h>
#include <random>
#include <unordered_set>
#include <nlohmann/json.hpp>
#include "engine.h"
#include "base_components.h"


std::string to_string(const std::vector<double>& v) {
    std::ostringstream oss;
    oss << "[";

    for (size_t i = 0; i < v.size(); ++i) {
        oss << v[i];
        if (i + 1 < v.size()) oss << ", ";
    }

    oss << "]";
    return oss.str();
}


class EngineTest : public ::testing::Test {
protected:
    std::unique_ptr<InitStateGenerator> init_generator;
    std::mt19937 rng;
    JANIEngine engine;
    void SetUp() override {
        rng = std::mt19937(50); // Fixed seed for reproducibility
        // Add some constants to the engine
        nlohmann::json constants_json = nlohmann::json::parse(R"([
            {
                "name": "gravity",
                "type": "real",
                "value": -9.8067
            },
            {
                "name": "timestep",
                "type": "real",
                "value": 0.3
            }
        ])");
        for (auto it = constants_json.begin(); it != constants_json.end(); ++it) {
            std::unique_ptr<Variable> constant = engine.testConstructConstant(*it);
            engine.testAddConstant(std::move(constant));
        }
        // Add some variables to the engine
        nlohmann::json variables_json = nlohmann::json::parse(R"([
            {
                "name": "episode",
                "type": {
                    "base": "int",
                    "lower-bound": 0,
                    "upper-bound": 4000
                },
                "initial-value": 0
            },
            {
                "name": "height",
                "type": {
                    "base": "real",
                    "lower-bound": 0,
                    "upper-bound": 1000
                },
                "initial-value": 5
            },
            {
                "name": "velocity",
                "type": {
                    "base": "real",
                    "lower-bound": -100,
                    "upper-bound": 100
                },
                "initial-value": 1
            }
        ])");
        for (auto it = variables_json.begin(); it != variables_json.end(); ++it) {
            std::unique_ptr<Variable> var = engine.testConstructVariable(*it);
            engine.testAddVariable(std::move(var));
        }
        // Add some initial states
        nlohmann::json json_obj = nlohmann::json::parse(R"([
            {
                "variables": [
                    {
                        "value": 0,
                        "var": "episode"
                    },
                    {
                        "value": 7.4787,
                        "var": "height"
                    },
                    {
                        "value": -0.651,
                        "var": "velocity"
                    }
                ]
            },
            {
                "variables": [
                    {
                        "value": 0,
                        "var": "episode"
                    },
                    {
                        "value": 8.0739,
                        "var": "height"
                    },
                    {
                        "value": 0.8914,
                        "var": "velocity"
                    }
                ]
            },
            {
                "variables": [
                    {
                        "value": 0,
                        "var": "episode"
                    },
                    {
                        "value": 7.1719,
                        "var": "height"
                    },
                    {
                        "value": 0.2349,
                        "var": "velocity"
                    }
                ]
            }
        ])");
        init_generator = engine.testConstructGeneratorFromValues(json_obj);
        // Add an automaton
        nlohmann::json automaton_json = nlohmann::json::parse(R"(
        {
            "edges": [
                {
                    "action": "push",
                    "destinations": [
                        {
                            "assignments": [
                                {
                                    "ref": "height",
                                    "value": {
                                        "left": "height",
                                        "op": "+",
                                        "right": {
                                            "left": {
                                                "left": "velocity",
                                                "op": "*",
                                                "right": "timestep"
                                            },
                                            "op": "+",
                                            "right": {
                                                "left": {
                                                    "left": 0.5,
                                                    "op": "*",
                                                    "right": "gravity"
                                                },
                                                "op": "*",
                                                "right": {
                                                    "left": "timestep",
                                                    "op": "*",
                                                    "right": "timestep"
                                                }
                                            }
                                        }
                                    }
                                },
                                {
                                    "ref": "velocity",
                                    "value": {
                                        "left": "velocity",
                                        "op": "+",
                                        "right": {
                                            "left": {
                                                "left": "timestep",
                                                "op": "*",
                                                "right": "gravity"
                                            },
                                            "op": "-",
                                            "right": 4
                                        }
                                    }
                                },
                                {
                                    "ref": "episode",
                                    "value": {
                                        "left": "episode",
                                        "op": "+",
                                        "right": 1
                                    }
                                }
                            ],
                            "location": "loc_0"
                        }
                    ],
                    "guard": {
                        "exp": {
                            "left": {
                                "left": 5,
                                "op": "≤",
                                "right": "height"
                            },
                            "op": "∧",
                            "right": {
                                "left": {
                                    "left": "height",
                                    "op": "≤",
                                    "right": 9
                                },
                                "op": "∧",
                                "right": {
                                    "left": {
                                        "left": {
                                            "left": {
                                                "left": {
                                                    "left": 0,
                                                    "op": "-",
                                                    "right": "velocity"
                                                },
                                                "op": "*",
                                                "right": "timestep"
                                            },
                                            "op": "-",
                                            "right": {
                                                "left": 0.5,
                                                "op": "*",
                                                "right": {
                                                    "left": "gravity",
                                                    "op": "*",
                                                    "right": {
                                                        "left": "timestep",
                                                        "op": "*",
                                                        "right": "timestep"
                                                    }
                                                }
                                            }
                                        },
                                        "op": "<",
                                        "right": "height"
                                    },
                                    "op": "∧",
                                    "right": {
                                        "left": {
                                            "left": -100,
                                            "op": "≤",
                                            "right": {
                                                "left": {
                                                    "left": "velocity",
                                                    "op": "-",
                                                    "right": 4
                                                },
                                                "op": "+",
                                                "right": {
                                                    "left": "gravity",
                                                    "op": "*",
                                                    "right": "timestep"
                                                }
                                            }
                                        },
                                        "op": "∧",
                                        "right": {
                                            "left": {
                                                "left": "episode",
                                                "op": "+",
                                                "right": 1
                                            },
                                            "op": "≤",
                                            "right": 4000
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "location": "loc_0"
                }
            ]
        }
        )");
        std::unique_ptr<Automaton> automaton = engine.testConstructAutomaton(automaton_json, 0);
        engine.testAddAutomaton(std::move(automaton));
        // Add an action
        std::unique_ptr<Action> action = engine.testConstructAction("push", 0);
        engine.testAddAction(std::move(action));
        // Set objective expression
        nlohmann::json objective_json = nlohmann::json::parse(R"(
        {
            "goal": {
                "exp": {
                    "left": 1000,
                    "op": "≤",
                    "right": "episode"
                },
                "op": "state-condition"
            },
            "op": "objective"
        }
        )");
        std::unique_ptr<Expression> objective_expr = engine.testConstructObjectiveExpression(objective_json);
        engine.testSetObjectiveExpression(std::move(objective_expr));
    }
};


TEST_F(EngineTest, InitialStateGeneration) {
    const int num_samples = 100;
    std::unordered_set<State, StateHasher> samples;
    for (int i = 0; i < num_samples; ++i) {
        const State *state = init_generator->generateInitialState(rng);
        samples.insert(*state);
    }
    // Check that all samples are among the defined initial states
    for (const auto& sample : samples) {
        bool found = false;
        for (const auto& init_state : init_generator->getInitialStatePool()) {
            if (sample == (*init_state)) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "Sampled state not found in initial states.";
    }

    EXPECT_TRUE(samples.size() > 1) << "Initial state generator is not producing diverse states.";
    EXPECT_TRUE(samples.size() <= 3) << "Initial state generator is producing too many states.";
}

TEST_F(EngineTest, StateFromVector) {
    State s;
    s.setVariable("episode", std::make_unique<IntVariable>(2, "episode", 0, 4000, 10));
    s.setVariable("height", std::make_unique<RealVariable>(3, "height", 0, 1000, 7.3567));
    s.setVariable("velocity", std::make_unique<RealVariable>(4, "velocity", -100, 100, 3.1578));
    s.setVariable("gravity", std::make_unique<RealConstant>(0, "gravity", -9.8067));
    s.setVariable("timestep", std::make_unique<RealConstant>(1, "timestep", 0.3));

    std::vector<double> vec = {-9.8067, 0.3, 10.0, 7.3567, 3.1578};
    State s_from_vec = engine.create_state_from_vector(vec);
    EXPECT_TRUE(s == s_from_vec) << "State constructed from vector does not match expected state.";

    std::vector<double> false_vec = {-9.8067, 0.3, 10.0, 7.3567, 3.1579};
    State s_false = engine.create_state_from_vector(false_vec);
    EXPECT_FALSE(s == s_false) << "States with different variable values are considered equal.";
}

TEST_F(EngineTest, ResetFromStateVector) {
    std::vector<double> vec = {-9.8067, 0.3, 25.0, 12.5, -4.0};
    std::vector<double> restored = engine.reset_from_state_vector(vec);

    EXPECT_EQ(restored, vec);
    EXPECT_EQ(engine.get_current_state_vector(), vec);
}

TEST_F(EngineTest, StateHash) {
    State state_1;
    state_1.setVariable("x", std::make_unique<RealVariable>(0, "x", -9.7, 9.7, 6.356856));
    state_1.setVariable("y", std::make_unique<RealVariable>(1, "y", -5.0, 5.0, -3.1415968));
    state_1.setVariable("z", std::make_unique<BooleanVariable>(2, "z", true));

    State state_2;
    state_2.setVariable("x", std::make_unique<RealVariable>(0, "x", -9.7, 9.7, 6.356856));
    state_2.setVariable("y", std::make_unique<RealVariable>(1, "y", -5.0, 5.0, -3.1415968));
    state_2.setVariable("z", std::make_unique<BooleanVariable>(2, "z", true));

    StateHasher hasher;
    size_t hash_1 = hasher(state_1);
    size_t hash_2 = hasher(state_2);
    EXPECT_EQ(hash_1, hash_2) << "Hashes of identical states should be equal.";

    // Test corner case real numbers
    state_1.setVariable("x", std::make_unique<RealVariable>(0, "x", -9.7, 9.7, 1.9999999999));
    state_1.setVariable("y", std::make_unique<RealVariable>(1, "y", -5.0, 5.0, -3.000000000001));
    state_1.setVariable("z", std::make_unique<BooleanVariable>(2, "z", false));

    state_2.setVariable("x", std::make_unique<RealVariable>(0, "x", -9.7, 9.7, 1.9999999999));
    state_2.setVariable("y", std::make_unique<RealVariable>(1, "y", -5.0, 5.0, -3.000000000001));
    state_2.setVariable("z", std::make_unique<BooleanVariable>(2, "z", false));

    hash_1 = hasher(state_1);
    hash_2 = hasher(state_2);
    EXPECT_EQ(hash_1, hash_2) << "Hashes of identical states should be equal.";

    // Slightly modify state 2
    state_2.setVariable("x", std::make_unique<RealVariable>(0, "x", -9.7, 9.7, 1.9999999998));
    hash_2 = hasher(state_2);
    EXPECT_NE(hash_1, hash_2) << "Hashes should differ for different states.";

    state_2.setVariable("x", std::make_unique<RealVariable>(0, "x", -9.7, 9.7, 2.0));
    hash_2 = hasher(state_2);
    EXPECT_NE(hash_1, hash_2) << "Hashes should differ for different states.";
}


TEST_F(EngineTest, StateHashInSet) {
    std::unordered_set<State, StateHasher> state_set;

    State state_1;
    state_1.setVariable("x", std::make_unique<RealVariable>(0, "x", -9.7, 9.7, 6.356856));
    state_1.setVariable("y", std::make_unique<RealVariable>(1, "y", -5.0, 5.0, -3.1415968));
    state_1.setVariable("z", std::make_unique<BooleanVariable>(2, "z", true));
    state_set.insert(state_1);
    // Construct an identical state
    State state_2;
    state_2.setVariable("x", std::make_unique<RealVariable>(0, "x", -9.7, 9.7, 6.356856));
    state_2.setVariable("y", std::make_unique<RealVariable>(1, "y", -5.0, 5.0, -3.1415968));
    state_2.setVariable("z", std::make_unique<BooleanVariable>(2, "z", true));
    EXPECT_TRUE(state_set.find(state_2) != state_set.end()) << "State set should contain identical state.";
    
    State state_3;
    state_3.setVariable("x", std::make_unique<RealVariable>(0, "x", -9.7, 9.7, 1.9999999999));
    state_3.setVariable("y", std::make_unique<RealVariable>(1, "y", -5.0, 5.0, -3.000000000001));
    state_3.setVariable("z", std::make_unique<BooleanVariable>(2, "z", false));
    state_set.insert(state_3);

    State state_4;
    state_4.setVariable("x", std::make_unique<RealVariable>(0, "x", -9.7, 9.7, 1.9999999999));
    state_4.setVariable("y", std::make_unique<RealVariable>(1, "y", -5.0, 5.0, -3.000000000001));
    state_4.setVariable("z", std::make_unique<BooleanVariable>(2, "z", false));
    EXPECT_TRUE(state_set.find(state_4) != state_set.end()) << "State set should contain identical state.";

    State state_5;
    state_5.setVariable("x", std::make_unique<RealVariable>(0, "x", -9.7, 9.7, 1.9999999998));
    state_5.setVariable("y", std::make_unique<RealVariable>(1, "y", -5.0, 5.0, -3.000000000001));
    state_5.setVariable("z", std::make_unique<BooleanVariable>(2, "z", false)); 
    EXPECT_TRUE(state_set.find(state_5) == state_set.end()) << "State set should not contain different state.";

    State state_6;
    state_6.setVariable("x", std::make_unique<RealVariable>(0, "x", -9.7, 9.7, 1.9999999999));
    state_6.setVariable("y", std::make_unique<RealVariable>(1, "y", -5.0, 5.0, -3.0));
    state_6.setVariable("z", std::make_unique<BooleanVariable>(2, "z", false));
    EXPECT_TRUE(state_set.find(state_6) == state_set.end()) << "State set should not contain different state.";
}

TEST_F(EngineTest, StateHashByPointer) {
    std::unordered_set<State, StateHasher> state_set;

    State *state_1 = new State();
    state_1->setVariable("x", std::make_unique<RealVariable>(0, "x", -9.7, 9.7, 6.356856));
    state_1->setVariable("y", std::make_unique<RealVariable>(1, "y", -5.0, 5.0, -3.1415968));
    state_1->setVariable("z", std::make_unique<BooleanVariable>(2, "z", true));
    state_set.insert(*state_1);
    // Construct an identical state
    State *state_2 = new State();
    state_2->setVariable("x", std::make_unique<RealVariable>(0, "x", -9.7, 9.7, 6.356856));
    state_2->setVariable("y", std::make_unique<RealVariable>(1, "y", -5.0, 5.0, -3.1415968));
    state_2->setVariable("z", std::make_unique<BooleanVariable>(2, "z", true));
    EXPECT_TRUE(state_set.find(*state_2) != state_set.end()) << "State set should contain identical state.";
    
    State *state_3 = new State();
    state_3->setVariable("x", std::make_unique<RealVariable>(0, "x", -9.7, 9.7, 1.9999999999));
    state_3->setVariable("y", std::make_unique<RealVariable>(1, "y", -5.0, 5.0, -3.000000000001));
    state_3->setVariable("z", std::make_unique<BooleanVariable>(2, "z", false));
    state_set.insert(*state_3);

    State *state_4 = new State();
    state_4->setVariable("x", std::make_unique<RealVariable>(0, "x", -9.7, 9.7, 1.9999999999));
    state_4->setVariable("y", std::make_unique<RealVariable>(1, "y", -5.0, 5.0, -3.000000000001));
    state_4->setVariable("z", std::make_unique<BooleanVariable>(2, "z", false));
    EXPECT_TRUE(state_set.find(*state_4) != state_set.end()) << "State set should contain identical state.";

    State *state_5 = new State();
    state_5->setVariable("x", std::make_unique<RealVariable>(0, "x", -9.7, 9.7, 1.9999999998));
    state_5->setVariable("y", std::make_unique<RealVariable>(1, "y", -5.0, 5.0, -3.000000000001));
    state_5->setVariable("z", std::make_unique<BooleanVariable>(2, "z", false)); 
    EXPECT_TRUE(state_set.find(*state_5) == state_set.end()) << "State set should not contain different state.";
    State *state_6 = new State();
    state_6->setVariable("x", std::make_unique<RealVariable>(0, "x", -9.7, 9.7, 1.9999999999));
    state_6->setVariable("y", std::make_unique<RealVariable>(1, "y", -5.0, 5.0, -3.0));
    state_6->setVariable("z", std::make_unique<BooleanVariable>(2, "z", false));
    EXPECT_TRUE(state_set.find(*state_6) == state_set.end()) << "State set should not contain different state.";
}

TEST_F(EngineTest, StepTest) {
    State s;
    s.setVariable("episode", std::make_unique<IntVariable>(2, "episode", 0, 4000, 10));
    s.setVariable("height", std::make_unique<RealVariable>(3, "height", 0, 1000, 7.0));
    s.setVariable("velocity", std::make_unique<RealVariable>(4, "velocity", -100, 100, 3.0));
    s.setVariable("gravity", std::make_unique<RealConstant>(0, "gravity", -9.8067));
    s.setVariable("timestep", std::make_unique<RealConstant>(1, "timestep", 0.3));

    engine.set_current_state(s);
    std::vector<double> target_state_vector = {-9.8067, 0.3, 11, 7.0 + 3.0 * 0.3 + 0.5 * -9.8067 * 0.3 * 0.3, 3.0 + (-9.8067 * 0.3 - 4)};
    std::vector<double> next_state_vector = engine.step(0); // Action index 0 corresponds to "push"
    for (size_t i = 0; i < next_state_vector.size(); ++i) {
        EXPECT_NEAR(next_state_vector[i], target_state_vector[i], 1e-10) << "Mismatch at index " << i << ": expected " << target_state_vector[i] << ", got " << next_state_vector[i];
    }
    // Check not equal
    std::vector<double> wrong_state_vector = {-9.8067, 0.3, 11, 7.4587, -3.0};
    EXPECT_FALSE(next_state_vector == wrong_state_vector) << "Next state vector should not match wrong state vector.";
    // EXPECT_TRUE(next_state_vector == target_state_vector) << "Next state vector " << to_string(next_state_vector) << " does not match expected values " << to_string(target_state_vector) << ".";

    std::vector<double> next_target_state_vector = {-9.8067, 0.3, 12, 
        (7.0 + 3.0 * 0.3 + 0.5 * -9.8067 * 0.3 * 0.3) + 
        ((3.0 + (-9.8067 * 0.3 - 4)) * 0.3 + 0.5 * -9.8067 * 0.3 * 0.3),
        (3.0 + (-9.8067 * 0.3 - 4)) + (-9.8067 * 0.3 - 4)};
    next_state_vector = engine.step(0); // Action index 0 corresponds to "push"
    for (size_t i = 0; i < next_state_vector.size(); ++i) {
        EXPECT_DOUBLE_EQ(next_state_vector[i], next_target_state_vector[i]) << "Mismatch at index " << i << ": expected " << next_target_state_vector[i] << ", got " << next_state_vector[i];
        // EXPECT_NEAR(next_state_vector[i], next_target_state_vector[i], 1e-10) << "Mismatch at index " << i << ": expected " << next_target_state_vector[i] << ", got " << next_state_vector[i];
    }
}

TEST_F(EngineTest, ConstructionTest) {
    JANIEngine engine_from_file = JANIEngine(
        "../../../examples/bouncing_ball/bouncing_ball.jani",
        "",
        "../../../examples/bouncing_ball/start.jani",
        "../../../examples/bouncing_ball/objective.jani",
        "../../../examples/bouncing_ball/safe.jani",
        42);
    EXPECT_EQ(engine_from_file.get_num_variables(), 3);
    EXPECT_EQ(engine_from_file.get_num_constants(), 8);
    EXPECT_EQ(engine_from_file.get_num_actions(), 2);
}

TEST_F(EngineTest, GoalConditionTest) {
    std::vector<double> goal_conditions = engine.extract_goal_condition();
    EXPECT_EQ(goal_conditions.size(), 1);
    EXPECT_DOUBLE_EQ(goal_conditions[0], 1000.0);
}

TEST_F(EngineTest, ConditionFromStateTest) {
    State s;
    s.setVariable("episode", std::make_unique<IntVariable>(2, "episode", 0, 4000, 10));
    s.setVariable("height", std::make_unique<RealVariable>(3, "height", 0, 1000, 7.0));
    s.setVariable("velocity", std::make_unique<RealVariable>(4, "velocity", -100, 100, 3.0));
    s.setVariable("gravity", std::make_unique<RealConstant>(0, "gravity", -9.8067));
    s.setVariable("timestep", std::make_unique<RealConstant>(1, "timestep", 0.3));

    std::vector<double> state_vector = s.toRealVector();
    std::vector<double> conditions = engine.extract_condition_from_state_vector(state_vector);
    EXPECT_EQ(conditions.size(), 1);
    EXPECT_LE(conditions[0], 10.0);
}

TEST_F(EngineTest, ComplexConditionsTest) {
    nlohmann::json objective_json = nlohmann::json::parse(R"(
    {
        "goal": {
            "exp": {
                "left": {
                    "left": 1000,
                    "op": "≤",
                    "right": "episode"
                },
                "op": "∧",
                "right": {
                    "left": "height",
                    "op": "=",
                    "right": 500
                }
            },
            "op": "state-condition"
        },
        "op": "objective"
    }
    )");
    std::unique_ptr<Expression> objective_expr = engine.testConstructObjectiveExpression(objective_json);
    engine.testSetObjectiveExpression(std::move(objective_expr));
    std::vector<double> goal_conditions = engine.extract_goal_condition();
    EXPECT_EQ(goal_conditions.size(), 2);
    EXPECT_DOUBLE_EQ(goal_conditions[0], 1000.0);
    EXPECT_DOUBLE_EQ(goal_conditions[1], 500.0);

    State s;
    s.setVariable("episode", std::make_unique<IntVariable>(2, "episode", 0, 4000, 10));
    s.setVariable("height", std::make_unique<RealVariable>(3, "height", 0, 1000, 7.0));
    s.setVariable("velocity", std::make_unique<RealVariable>(4, "velocity", -100, 100, 3.0));
    s.setVariable("gravity", std::make_unique<RealConstant>(0, "gravity", -9.8067));
    s.setVariable("timestep", std::make_unique<RealConstant>(1, "timestep", 0.3));
    std::vector<double> state_vector = s.toRealVector();
    std::vector<double> conditions = engine.extract_condition_from_state_vector(state_vector);
    EXPECT_EQ(conditions.size(), 2);
    EXPECT_LE(conditions[0], 10.0);
    EXPECT_DOUBLE_EQ(conditions[1], 7.0);
}

class EngineSimpleAutomatonTest : public ::testing::Test {
protected:
    JANIEngine engine;

    void SetUp() override {
        nlohmann::json automaton_json = nlohmann::json::parse(R"({
            "edges": [
            {
                "action": "a",
                "guard": { 
                    "exp": {
                        "left": "x",
                        "right": 5,
                        "op": "<"
                    }
                },
                "destinations": [
                    {
                        "assignments": [
                            { "ref": "x", 
                              "value":  
                                {
                                    "left": "x",
                                    "right": 1,
                                    "op": "+"
                                }
                            }
                        ]
                    }
                ]
            },
            {
                "action": "b",
                "guard": { 
                    "exp": {
                        "left": {
                            "left": 10,
                            "right": "x",
                            "op": "<"
                        },
                        "right": {
                            "left": 7,
                            "right": "y",
                            "op": "<"
                        },
                        "op": "∧"
                    }
                },
                "destinations": [
                    {
                        "assignments": [
                            { "ref": "x", 
                              "value":  
                                {
                                    "left": "x",
                                    "right": 2,
                                    "op": "-"
                                }
                            },
                            { "ref": "y",
                              "value":
                                {
                                    "left": "y",
                                    "right": 3,
                                    "op": "*"
                                }
                            }
                        ]
                    }
                ]
            }]
        })");
        std::unique_ptr<Automaton> automaton = engine.testConstructAutomaton(automaton_json, 0);
        engine.testAddAutomaton(std::move(automaton));
        // Action a for testing
        std::unique_ptr<Action> action_a = engine.testConstructAction("a", 0);
        engine.testAddAction(std::move(action_a));
        // Action b for testing
        std::unique_ptr<Action> action_b = engine.testConstructAction("b", 1);
        engine.testAddAction(std::move(action_b));
    }
};

TEST_F(EngineSimpleAutomatonTest, ActionMaskTest) {
    State s;
    s.setVariable("x", std::make_unique<IntVariable>(0, "x", 0, 50, 3)); // x = 3
    s.setVariable("y", std::make_unique<IntVariable>(1, "y", 0, 50, 9)); // y = 9

    std::vector<bool> action_mask = engine.get_action_mask(s);
    EXPECT_EQ(action_mask.size(), 2); // Should contain 2 actions
    EXPECT_TRUE(action_mask[0]); // Action a should be applicable
    EXPECT_FALSE(action_mask[1]); // Action b should not be applicable
}
