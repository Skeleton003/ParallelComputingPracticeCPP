enable_language(CXX)
# 批量的自动查找所有的.cpp 和 .h 文件
file(GLOB include_sources CONFIGURE_DEPENDS include/*.cpp include/*.h)

# 将include设置为动态库
# 当dll和exe不在同一目录时，只会查找exe所在目录和PATH
add_library(include SHARED ${include_sources})
set_property(TARGET include PROPERTY RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set_property(TARGET include PROPERTY ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set_property(TARGET include PROPERTY LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set_property(TARGET include PROPERTY RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR})
set_property(TARGET include PROPERTY ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR})
set_property(TARGET include PROPERTY LIBRARY_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR})
set_property(TARGET include PROPERTY RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR})
set_property(TARGET include PROPERTY ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR})
set_property(TARGET include PROPERTY LIBRARY_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR})


target_compile_definitions(include PRIVATE include_EXPORT)
#设置额外的头文件查找目录
target_include_directories(include PUBLIC .)