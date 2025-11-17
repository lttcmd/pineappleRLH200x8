set(SRC
  ofc_cpp.cpp
  ofc_score.cpp
  ofc_step.cpp
  ofc_actions.cpp
  ofc_encode.cpp
  ofc_engine.cpp
)

# Require C++20 for designated initializers used in Rules.hpp
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

pybind11_add_module(ofc_cpp ${SRC})
target_include_directories(ofc_cpp PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include
)
target_compile_definitions(ofc_cpp PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES)

# Install the extension at top-level (so import ofc_cpp works)
install(TARGETS ofc_cpp
  LIBRARY DESTINATION .
  ARCHIVE DESTINATION .
  RUNTIME DESTINATION .
)


