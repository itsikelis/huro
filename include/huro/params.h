// Copyright 2025 Ioannis Tsikelis

#ifndef HURO_PARAMS_H_
#define HURO_PARAMS_H_

#include <cstddef>
#include <string>
#include <vector>

namespace huro {
struct Params {
  bool high_freq;
  bool info_imu;
  bool info_motors;
  size_t n_motors;
  std::vector<std::string> joint_names;
};

} // namespace huro
#endif // HURO_PARAMS_H_
