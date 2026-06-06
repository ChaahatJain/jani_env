#ifndef DOMAIN_SPECIFIC_H
#define DOMAIN_SPECIFIC_H

#include <string>
#include <memory>
#include <stdexcept>
#include "../search/states/state_values.h"
#include "../search/using_search.h"
#include "../search/information/jani_2_interface.h"
#include "../parser/ast/model.h"
#include <iostream>

// Forward declarations - replace these with actual includes for your project.
// These types are assumed to exist elsewhere in your codebase.
// class StateValues;

// =====================================================================
// Abstract base class
// =====================================================================
class DomainSpecific {
public:
    virtual ~DomainSpecific() = default;

    DomainSpecific(const Model* model, const Jani2Interface* jani)
        : model_(model), jani_(jani) {}

    // The single pure-virtual interface every domain policy must implement.
    virtual ActionLabel_type get_action(const StateValues& state_values) = 0;

    // Factory: build the correct concrete policy from a string name.
    // Throws std::invalid_argument if the name is unknown.
    static std::unique_ptr<DomainSpecific> create(const std::string& name, const Model* model, const Jani2Interface* jani);

protected: 
    const Model* model_;
    const Jani2Interface* jani_;
};

// =====================================================================
// Derived classes
// =====================================================================
class Blocksworld : public DomainSpecific {
public:
    using DomainSpecific::DomainSpecific;

    ActionLabel_type get_action(const StateValues& state_values) override;
};

class Transport : public DomainSpecific {
public:
    using DomainSpecific::DomainSpecific;

    ActionLabel_type get_action(const StateValues& state_values) override;
};

class OneWayLine : public DomainSpecific {
public:
    using DomainSpecific::DomainSpecific;
    ActionLabel_type get_action(const StateValues& state_values) override;
};

class OneWayLinePark : public DomainSpecific {
public:
    using DomainSpecific::DomainSpecific;
    ActionLabel_type get_action(const StateValues& state_values) override;
};

class TwoWayLine : public DomainSpecific {
public:
    using DomainSpecific::DomainSpecific;
    ActionLabel_type get_action(const StateValues& state_values) override;
};

class TwoWayLinePark : public DomainSpecific {
public:
    using DomainSpecific::DomainSpecific;
    ActionLabel_type get_action(const StateValues& state_values) override;
};

class Transport_Feat : public DomainSpecific {
public:
    using DomainSpecific::DomainSpecific;
    ActionLabel_type get_action(const StateValues& state_values) override;
};

class BouncingBall : public DomainSpecific {
public:
    using DomainSpecific::DomainSpecific;
    ActionLabel_type get_action(const StateValues& state_values) override;
};

class FollowCar : public DomainSpecific {
public:
    using DomainSpecific::DomainSpecific;
    ActionLabel_type get_action(const StateValues& state_values) override;
};

class InvertedPendulum : public DomainSpecific {
public:
    using DomainSpecific::DomainSpecific;
    ActionLabel_type get_action(const StateValues& state_values) override;
};

class Cartpole : public DomainSpecific {
public:
    using DomainSpecific::DomainSpecific;
    ActionLabel_type get_action(const StateValues& state_values) override;
};

class Beluga: public DomainSpecific {
public:
    using DomainSpecific::DomainSpecific;
    ActionLabel_type get_action(const StateValues& state_values) override;
};

#endif // DOMAIN_SPECIFIC_H