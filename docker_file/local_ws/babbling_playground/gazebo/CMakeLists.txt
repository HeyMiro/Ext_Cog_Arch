cmake_minimum_required(VERSION 3.0.2)
project(babbling_example)

find_package(catkin REQUIRED)
catkin_package()

catkin_install_python(PROGRAMS
  scripts/datasets.py
  scripts/mirror_babbling.py
  scripts/neural_net.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY assets launch meshes urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
