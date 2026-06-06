#include "domain_specific.h"
std::unique_ptr<DomainSpecific> DomainSpecific::create(const std::string& name, const Model* model, const Jani2Interface* jani) {
    std::cout << "Constructing manual policy for: " << name << std::endl;
    if (name == "Blocksworld")       return std::make_unique<Blocksworld>(model, jani);
    if (name == "Transport")         return std::make_unique<Transport>(model, jani);
    if (name == "OneWayLine")        return std::make_unique<OneWayLine>(model, jani);
    if (name == "OneWayLinePark")    return std::make_unique<OneWayLinePark>(model, jani);
    if (name == "TwoWayLine")        return std::make_unique<TwoWayLine>(model, jani);
    if (name == "TwoWayLinePark")    return std::make_unique<TwoWayLinePark>(model, jani);
    if (name == "Transport_Feat")    return std::make_unique<Transport_Feat>(model, jani);
    if (name == "BouncingBall")      return std::make_unique<BouncingBall>(model, jani);
    if (name == "FollowCar")         return std::make_unique<FollowCar>(model, jani);
    if (name == "InvertedPendulum")  return std::make_unique<InvertedPendulum>(model, jani);
    if (name == "Cartpole")          return std::make_unique<Cartpole>(model, jani);
    if (name == "Beluga")            return std::make_unique<Beluga>(model, jani);

    throw std::invalid_argument("Unknown domain-specific policy: " + name);
}