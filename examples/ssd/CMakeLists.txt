find_package(OpenCV REQUIRED core highgui )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../../src)



include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(ssdmobilenet ssdmobilenet.cpp)
target_link_libraries(ssdmobilenet ncnn ${OpenCV_LIBS})

add_executable(ssdsqueezenet ssdsqueezenet.cpp)
target_link_libraries(ssdsqueezenet ncnn ${OpenCV_LIBS})

IF(WIN32)
	set(CMAKE_CXX_FLAGS_RELEASE "/MT")
    set(CMAKE_CXX_FLAGS_DEBUG "/MTd")
ENDIF(WIN32)