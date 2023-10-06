cmake_minimum_required(VERSION 3.12)
project(YoloNAS)

set(CMAKE_CXX_STANDARD 11)
add_library(YoloNAS src/YoloNAS.cpp)

target_include_directories(YoloNAS PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

find_package(OpenCV REQUIRED)
target_link_libraries(YoloNAS PRIVATE ${OpenCV_LIBS})

install(TARGETS YoloNAS
    EXPORT YoloNASTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/ DESTINATION include)

install(EXPORT YoloNASTargets
    FILE YoloNASTargets.cmake
    NAMESPACE ukicomputers::
    DESTINATION lib/cmake/YoloNAS
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    YoloNASConfigVersion.cmake
    VERSION 1.0
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/YoloNASConfig.cmake.in
    YoloNASConfig.cmake
    INSTALL_DESTINATION lib/cmake/YoloNAS
)

install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/YoloNASConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/YoloNASConfigVersion.cmake
    DESTINATION lib/cmake/YoloNAS
)