#version 460

// === CPU/GPU interface ===
//
// Any modification to the following declarations may call for matching
// modifications to the CPU-side code!

// Workgroup dimensions
layout(local_size_x = 8, local_size_y = 8) in;

// Initial concentrations of species U and V
layout(set = 0, binding = 0) uniform sampler2D us;
layout(set = 0, binding = 1) uniform sampler2D vs;

// Final concentrations of species U and V
layout(set = 0, binding = 2) uniform writeonly image2D nextUs;
layout(set = 0, binding = 3) uniform writeonly image2D nextVs;

// Computation parameters
layout(set = 1, binding = 0) uniform Parameters {
    mat3 weights;
    float diffusion_rate_u, diffusion_rate_v,
          feed_rate, kill_rate,
          time_step;
} params;

// Diffusion stencil properties
const ivec2 stencil_shape = ivec2(3, 3);
const ivec2 stencil_offset = ivec2((stencil_shape - 1) / 2);

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

// What each shader invocation does
void main() {
    // Determine which input and output location we act on
    const ivec2 center_idx = ivec2(gl_GlobalInvocationID.xy);

    // Read the center value of the concentration
    const vec2 uv = read_uv(center_idx);

    // Compute diffusion term for U and V
    vec2 full_uv = vec2(0.);
    const ivec2 top_left = center_idx - stencil_offset;
    for (int x = 0; x < stencil_shape.x; ++x) {
        for (int y = 0; y < stencil_shape.y; ++y) {
            const vec2 stencil_uv = read_uv(top_left + ivec2(x, y));
            full_uv += params.weights[x][y] * (stencil_uv - uv);
        }
    }
    const vec2 diffusion_rate = vec2(params.diffusion_rate_u, params.diffusion_rate_v);
    const vec2 diffusion = diffusion_rate * full_uv;

    // Deduce rate of change in u and v
    const float u = uv.x;
    const float diffusion_u = diffusion.x;
    const float v = uv.y;
    const float diffusion_v = diffusion.y;
    //
    const float uv_square = u * v * v;
    const float du = diffusion_u - uv_square + params.feed_rate * (1.0 - u);
    const float dv = diffusion_v + uv_square - (params.feed_rate + params.kill_rate) * v;

    // Update u and v accordingly
    write_uv(center_idx, uv + vec2(du, dv) * params.time_step);
}