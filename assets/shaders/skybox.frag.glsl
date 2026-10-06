#version 430

in vec3 v_dir;
flat in vec3 sunPosRot;
out vec4 fragColor;

uniform float skyBrightness;
uniform float duskWeight;
uniform vec3  duskColor;

layout (binding = 4) uniform samplerCube skyTex;

void main() {
    vec3 d = normalize(v_dir);

    // -------- 太阳方向性亮度 --------
    float dayFactor = smoothstep(-0.15, 0.30, normalize(sunPosRot).y);

    float cosSun = max(dot(d.yz, normalize(sunPosRot).yz), 0.0);

    float dirBoost = 1.0 + 0.4 * pow(cosSun, 5.0) * dayFactor;

    vec3 col = texture(skyTex, d).rgb * skyBrightness * dirBoost;

    // -------- 黄昏地平线光晕 --------
    float horiz = 1.0 - smoothstep(0.0, 0.3, abs(d.y));
    horiz = pow(horiz, 1.6);

    vec2 xz = d.xz;
    float l2 = dot(xz, xz);
    vec2 myXZ = (l2 > 1e-8) ? xz * inversesqrt(l2) : vec2(0.0);
    float azDot    = dot(myXZ, normalize(sunPosRot.xz));
    float sunAlign = clamp(0.5 + 0.5 * azDot, 0.0, 1.0);
    sunAlign = sunAlign * sunAlign;

    float glow = horiz * duskWeight * (0.25 + 0.75 * sunAlign);
    col += duskColor * glow * 0.6;

    fragColor = vec4(col, 1.0);
}