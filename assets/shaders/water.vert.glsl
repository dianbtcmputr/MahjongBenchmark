#version 430
layout(location = 0) in vec3 position;

uniform mat4 proj;
uniform mat4 view;

out vec3 v_world;

void main() {
    v_world = position;
    gl_Position = proj * view * vec4(position, 1.0);
}