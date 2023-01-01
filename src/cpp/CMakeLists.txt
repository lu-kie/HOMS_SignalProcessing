# CMakeList.txt : CMake project for HOMS, include source and define
# project specific logic here.
#

# Add source to this project's executable.
project(HOMS)

add_subdirectory("eigen-3.4.0")
include_directories("eigen-3.4.0")

find_package(Eigen3 REQUIRED)

add_executable(
#HOMS "HOMS.cpp" "HigherOrderMS_1D.cpp"
HOMStests "HOMStests.cpp" "HigherOrderMS_1D.cpp" "Interval.cpp" "PcwSmoothPartitioningBase.cpp" "PcwPolynomialPartitioning.cpp" "PcwSmoothPartitioning.cpp")


target_link_libraries(
HOMStests PUBLIC
Eigen3::Eigen
GTest::gtest_main
)


if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET HOMStests PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
