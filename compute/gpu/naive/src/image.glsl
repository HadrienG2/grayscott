// Commonalities between image-based Gray-Scott computations

// === CPU/GPU interface ===
//
// Any modification to the following declarations may call for matching
// modifications to the CPU-side code.
// Adding supplementary descriptor sets will require changes to code
// including this header to avoid collisions.

// Initial concentrations of species U and V
layout(set = 0, binding = 0) uniform sampler2D us;
layout(set = 0, binding = 1) uniform sampler2D vs;

// Final concentrations of species U and V
layout(set = 0, binding = 2) uniform restrict writeonly image2D nextUs;
layout(set = 0, binding = 3) uniform restrict writeonly image2D nextVs;

// === GPU-side code ===

// Read the current value of the concentration
vec2 read_uv(const ivec2 input_idx) {
    const vec2 input_pos = vec2(input_idx) + 0.5;
    return vec2(
        texture(us, input_pos).r,
        texture(vs, input_pos).r
    );
}

// Write the next value of the concentration
void write_uv(const ivec2 output_idx, const vec2 uv) {
    imageStore(nextUs, output_idx, vec4(uv.x));
    imageStore(nextVs, output_idx, vec4(uv.y));
}