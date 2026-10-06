#version 430
layout(location = 0) in vec3 position;

uniform mat4  proj;
uniform mat4  view;
uniform vec3  halfSize;
uniform float radius;
uniform int   instanceOffset;

layout(std430, binding = 2) buffer ModelBuffer {
    mat4 models[];
};
layout(std430, binding = 3) buffer IdBuffer {
    uint objectIds[];
};
layout(std430, binding = 4) buffer LayerBuffer {
    uint texLayers[];
};

out vec3 v_position;
out vec3 v_local;
out vec3 v_world;
flat out mat4 v_model;
flat out uint v_objectIndex;
flat out uint v_texLayer;

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

    v_position    = rounded;
    v_local       = position;
    v_world       = (model * vec4(rounded, 1.0)).xyz;
    v_model       = model;
    v_objectIndex = objectIds[inst];
    v_texLayer    = texLayers[inst];

    gl_Position = proj * view * model * vec4(rounded, 1.0);
}