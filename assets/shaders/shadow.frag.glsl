#version 430
in float v_linearDepth;
void main() {
    gl_FragDepth = clamp(v_linearDepth, 0.0, 1.0);
}