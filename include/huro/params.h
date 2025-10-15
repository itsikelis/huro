// Copyright 2025 Ioannis Tsikelis

#ifndef HURO_PARAMS_H_
#define HURO_PARAMS_H_

#include <cstddef>
#include <string>
#include <vector>

namespace huro {
struct Params {
  bool fix_base;
  bool info_imu;
  bool info_motors;
  size_t n_motors;
  int sim_dt_ms;
  std::string xml_filename;
  std::vector<std::string> joint_names;
  std::string lowstate_topic_name;
  std::string lowcmd_topic_name;
  std::string odom_topic_name;
  std::string base_link_name;
  std::string sole_link_name;
};

} // namespace huro
#endif // HURO_PARAMS_H_
