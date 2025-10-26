#pragma once

#include "network.h"

#include <fstream>

namespace Ember {
    void saveParams(const std::string& path, const Network& net);
    void loadParams(const std::string& path, Network& net);
}