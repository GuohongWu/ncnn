# Install script for directory: D:/haomaiyi/ncnn-win/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "D:/haomaiyi/ncnn-win/build_vs2015/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/haomaiyi/ncnn-win/build_vs2015/src/Debug/ncnn.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/haomaiyi/ncnn-win/build_vs2015/src/Release/ncnn.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/haomaiyi/ncnn-win/build_vs2015/src/MinSizeRel/ncnn.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/haomaiyi/ncnn-win/build_vs2015/src/RelWithDebInfo/ncnn.lib")
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "D:/haomaiyi/ncnn-win/src/blob.h"
    "D:/haomaiyi/ncnn-win/src/cpu.h"
    "D:/haomaiyi/ncnn-win/src/layer.h"
    "D:/haomaiyi/ncnn-win/src/layer_type.h"
    "D:/haomaiyi/ncnn-win/src/mat.h"
    "D:/haomaiyi/ncnn-win/src/modelbin.h"
    "D:/haomaiyi/ncnn-win/src/net.h"
    "D:/haomaiyi/ncnn-win/src/opencv.h"
    "D:/haomaiyi/ncnn-win/src/paramdict.h"
    "D:/haomaiyi/ncnn-win/src/benchmark.h"
    "D:/haomaiyi/ncnn-win/build_vs2015/src/layer_type_enum.h"
    "D:/haomaiyi/ncnn-win/build_vs2015/src/platform.h"
    )
endif()

