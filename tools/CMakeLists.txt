
IF(WIN32)
    SET (3DPARTY_PATH "D:/haomaiyi/ncnn-win/3rdparty/protobuf-3.0.2/cmake/build/protobuf")
	SET (PROTOBUF_INCLUDE_DIR ${3DPARTY_PATH}/include)
	SET (PROTOBUF_lib_dir ${3DPARTY_PATH}/lib)
	LINK_DIRECTORIES(${PROTOBUF_lib_dir})
	LIST(APPEND PROTOBUF_LIBRARY libprotobuf.lib)
	
	set(CMAKE_CXX_FLAGS_RELEASE "/MT")
    set(CMAKE_CXX_FLAGS_DEBUG "/MTd")
ENDIF(WIN32)

add_subdirectory(caffe)
add_subdirectory(mxnet)
add_subdirectory(onnx)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)

include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(ncnn2mem ncnn2mem.cpp)

target_link_libraries(ncnn2mem ncnn)
