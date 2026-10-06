#version 430
#define SKY_ROTATE_SPEED %SKY_ROTATE_SPEED%

layout(location = 0) in vec3 position;

uniform mat4  proj;
uniform mat4  view;
uniform float water_y;
uniform float poolHalfSize;
uniform float time;
uniform vec3  sunPos;

out vec3 v_dir;
flat out vec3 sunPosRot;

mat4 rotationY(float angle) {
    float c = cos(angle);
    float s = sin(angle);

    return mat4(
        c,   0.0, -s,   0.0,
        0.0, 1.0,  0.0, 0.0,
        s,   0.0,  c,   0.0,
        0.0, 0.0,  0.0, 1.0
    );
}

void main() {
    v_dir = normalize(position);

    sunPosRot = mat3(rotationY( - time * SKY_ROTATE_SPEED)) * sunPos;

    vec4 world = rotationY(time * SKY_ROTATE_SPEED) * vec4(
        position.x * poolHalfSize,
        water_y + position.y * poolHalfSize,
        position.z * poolHalfSize,
        1.0
    );
    gl_Position = proj * view * world;
}