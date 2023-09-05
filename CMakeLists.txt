cmake_minimum_required(VERSION 3.25)

# --- Specify C++ standard --- #
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- Create project and specify source files --- #
project(Regularization)
add_executable(regularization main.cpp regularizer.hpp)

# --- Find Vulkan installation --- #
find_package(Vulkan REQUIRED COMPONENTS glslc)
target_link_libraries(regularization PRIVATE Vulkan::Vulkan)

# --- Find shader files --- #
set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)

file(GLOB shader_files
  ${SHADER_SOURCE_DIR}/*.vert
  ${SHADER_SOURCE_DIR}/*.frag
  ${SHADER_SOURCE_DIR}/*.comp
)

# --- Create shader directory --- #
add_custom_command(
  COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_BINARY_DIR}
  OUTPUT ${SHADER_BINARY_DIR}
  COMMENT "Creating ${SHADER_BINARY_DIR}..."
)

# --- Add commands for compiling shaders --- #
foreach(shader_file IN LISTS shader_files)
	get_filename_component(FILENAME ${shader_file} NAME)
	add_custom_command(
		COMMAND ${Vulkan_GLSLC_EXECUTABLE} -o ${SHADER_BINARY_DIR}/${FILENAME}.spv ${shader_file}
		OUTPUT ${SHADER_BINARY_DIR}/${FILENAME}.spv
		DEPENDS ${shader_file} ${SHADER_BINARY_DIR}
		COMMENT "Compiling ${FILENAME}..."
	)
	list(APPEND spirv_shaders ${SHADER_BINARY_DIR}/${FILENAME}.spv)
endforeach()

# --- Add project dependency on shader files --- #
add_custom_target(shaders ALL DEPENDS ${spirv_shaders})
add_dependencies(regularization shaders)