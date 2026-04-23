idf_component_register(
    SRCS "main.c" "small_ae.c" "gmm.c" "knn.c"
    INCLUDE_DIRS "."
)

# Keep the hot loops simple and loop-unroll-friendly.
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -ffast-math)

# GCC can't prove runtime-K loop writes log_probs[0] before use.
set_source_files_properties(gmm.c PROPERTIES COMPILE_OPTIONS -Wno-maybe-uninitialized)
