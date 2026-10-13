add_executable(searchbenchmark searchbenchmark.cpp)
set_property(TARGET searchbenchmark PROPERTY CXX_STANDARD 20)
set_property(TARGET searchbenchmark PROPERTY CXX_STANDARD_REQUIRED ON)
target_include_directories(searchbenchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../find)

target_link_libraries(searchbenchmark PUBLIC simdutf::benchmarks::benchmark)
