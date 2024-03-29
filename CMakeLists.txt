cmake_minimum_required(VERSION 3.0.0)

project(HaxBallAgentGroup-2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math")

find_package(Qt5 COMPONENTS Widgets Gui REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(
    ../common/include/
    ../haxballenv/
    ../haxballenv/include
    ../haxballenv/include/Eigen)

set(SRC_FILES
    main.cpp
    ../common/src/ActionSpace.cpp
    ../common/src/RewardFunctions.cpp
    ../haxballenv/src/BaseAgent.cpp
    ../haxballenv/src/DummyAgent.cpp
    EvaluationCenter.cpp
    ../haxballenv/src/HaxBall.cpp
    ../haxballenv/src/HaxBallGui.cpp
    Peashooter.cpp
    ../haxballenv/src/RandomSearch.cpp)

set(MOC_FILES
    ../haxballenv/include/HaxBall.h
    ../haxballenv/include/HaxBallGui.h)

qt5_wrap_cpp(SRC_FILES ${MOC_FILES})

add_executable(${PROJECT_NAME} ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} Qt5::Widgets Qt5::Gui OpenMP::OpenMP_CXX)
