#version 430
layout(location = 0) in vec3 position;

uniform mat4  lightView;
uniform mat4  lightProj;
uniform vec3  halfSize;
uniform float radius;
uniform int   instanceOffset;
uniform vec3  lightPos;
uniform float lightFar;

layout(std430, binding = 2) buffer ModelBuffer {
    mat4 models[];
};

out float v_linearDepth;

void main() {
    int  inst  = gl_InstanceID + instanceOffset;
    mat4 model = models[inst];

    vec3  s        = halfSize;
    float rMax     = min(min(s.x, s.y), s.z) * 0.999;
    float r        = clamp(radius, 0.0, rMax);
    vec3  coreHalf = s - vec3(r);
    vec3  core     = clamp(position, -coreHalf, coreHalf);
    vec3  d        = position - core;
    float len      = length(d);
    vec3  rounded  = (len > 1e-5) ? core + d / len * r : position;

    vec4 world = model * vec4(rounded, 1.0);
    v_linearDepth = length(world.xyz - lightPos) / lightFar;

    gl_Position = lightProj * lightView * world;
}