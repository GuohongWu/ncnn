
IF(WIN32)
	SET (3DPARTY_PATH "D:/haomaiyi/ncnn-win/3rdparty")
	SET (OpenCV_DIR ${3DPARTY_PATH}/opencv/build)
ENDIF(WIN32)

find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
if(NOT OpenCV_FOUND)
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)
include_directories(${OpenCV_INCLUDE_DIRS})
LINK_DIRECTORIES(${OpenCV_LIB_DIR})

add_executable(squeezenet squeezenet.cpp)
target_link_libraries(squeezenet ncnn ${OpenCV_LIBS})

add_executable(fasterrcnn fasterrcnn.cpp)
target_link_libraries(fasterrcnn ncnn ${OpenCV_LIBS})

add_subdirectory(ssd)
