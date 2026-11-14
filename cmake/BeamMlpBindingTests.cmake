function(beam_add_mlp_binding_tests)
    get_filename_component(beam_binding_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
    add_executable(mlp_artifact_binding_cpu_tests
        "${beam_binding_root}/tests/mlp_artifact_binding_cpu.cpp"
        "${beam_binding_root}/src/native_input.cpp")
    target_compile_features(mlp_artifact_binding_cpu_tests PRIVATE cxx_std_17)
    # This self-contained test specialization must agree across both source
    # files; it deliberately does not link a differently sized beam_core.
    target_compile_definitions(mlp_artifact_binding_cpu_tests PRIVATE
        BEAM_STATE_LOGICAL_BYTES=8 BEAM_STATE_PHYSICAL_BYTES=16
        BEAM_STATE_ALIGNMENT=16 BEAM_MOVE_COUNT=3)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME mlp_artifact_binding_cpu
            COMMAND "${Python3_EXECUTABLE}"
                "${beam_binding_root}/tests/mlp_artifact_binding_cpu.py"
                $<TARGET_FILE:mlp_artifact_binding_cpu_tests>
                --tmp "${CMAKE_CURRENT_BINARY_DIR}/test_results/mlp_artifact_binding_cpu")
        set_tests_properties(mlp_artifact_binding_cpu PROPERTIES LABELS "host" TIMEOUT 120)
    else()
        message(WARNING "Python3 not found; mlp_artifact_binding_cpu CTest is unavailable")
    endif()
endfunction()
