#version 430
#define PCF_SAMPLES %PCF_SAMPLES%

in vec3 v_position;
in vec3 v_local;
in vec3 v_world;
flat in mat4 v_model;
flat in uint v_objectIndex;
flat in uint v_texLayer;
out vec4 fragColor;

// ---------- 材质 ----------
uniform vec3  baseColor_white;
uniform vec3  baseColor_green;
uniform vec3  halfSize;
uniform float radius;
uniform float carveDepth;

// ---------- 光照 ----------
uniform vec3  lightPos;
uniform vec3  camPos;
uniform vec3  lightColor;
uniform float lightIntensity;
uniform float ambient;
uniform float shininess;

// ---------- 贴图 ----------
// 纹理单元 0：麻将图案数组（多张等尺寸）
layout(binding = 0) uniform sampler2DArray mjTexArray;
// 纹理单元 1：桌面（单张，独立尺寸）
layout(binding = 1) uniform sampler2D tableTex;

uniform int useTableTex;
uniform int patternFace;

// ---------- 阴影 ----------
layout(binding = 2) uniform samplerCube shadowCube;
uniform int   shadowMode;
uniform float pcfRadius;
uniform float shadowBias;
uniform float shadowFar;

// ---------- 拾取 ----------
layout(std430, binding = 1) buffer PickBuffer {
    uint pickPacked;
};
uniform ivec2 mousePixel;
uniform int   highlightId;

vec4 samplePattern(vec2 uv) {
    if (useTableTex == 1) return texture(tableTex, uv);
    return texture(mjTexArray, vec3(uv, float(v_texLayer)));
}

ivec2 patternSize() {
    if (useTableTex == 1) return textureSize(tableTex, 0);
    return textureSize(mjTexArray, 0).xy;
}

float heightAt(vec2 uv) {
    vec3 c = samplePattern(uv).rgb;
    return min(min(c.r, c.g), c.b);
}

float pcfSamples(vec3 dn, vec3 rt, vec3 up2,
                 float currentDepth, float bias, float radius) {
    const float GOLDEN = 2.39996322972865332;   // π(3-√5)
    const int N = PCF_SAMPLES;

    float ang = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453)
              * 6.2831853;
    float ca = cos(ang), sa = sin(ang);

    float sum = 0.0;
    for (int i = 0; i < N; ++i) {
        float fi = float(i) + 0.5;
        float r  = sqrt(fi / float(N));
        float th = fi * GOLDEN;
        vec2  p  = r * vec2(cos(th), sin(th));

        vec2  rp  = vec2(p.x * ca - p.y * sa, p.x * sa + p.y * ca);
        vec3  off = (rp.x * rt + rp.y * up2) * radius;
        vec3  s   = normalize(dn + off);

        float d = texture(shadowCube, s).r;
        sum += (currentDepth - bias > d) ? 1.0 : 0.0;
    }
    return sum / float(N);
}

float computeShadow(vec3 nWorld) {
    if (shadowMode == 0) return 0.0;

    vec3  dir          = v_world - lightPos;
    float currentDepth = length(dir) / shadowFar;

    float nDotL = max(dot(nWorld, normalize(lightPos - v_world)), 0.0);
    // 斜率缩放偏置：掠射面放宽，消除硬阴影/CHS 的痤疮
    //float slope = sqrt(max(1.0 - nDotL * nDotL, 0.0)) / max(nDotL, 0.05);
    //float bias  = shadowBias * (1.0 + slope);

    float slope = min(sqrt(max(1.0 - nDotL*nDotL, 0.0)) / max(nDotL, 0.15), 3.0);
    float bias  = shadowBias * (1.0 + slope);

    // ---- 硬阴影 ----
    if (shadowMode == 1) {
        float d = texture(shadowCube, dir).r;
        return (currentDepth - bias > d) ? 1.0 : 0.0;
    }

    // ---- 以光方向为 z 轴构造切向基 ----
    vec3 dn  = normalize(dir);
    vec3 up  = abs(dn.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 rt  = normalize(cross(up, dn));
    vec3 up2 = cross(dn, rt);

    // ---- 固定半径 PCF ----
    if (shadowMode == 2) {
        return pcfSamples(dn, rt, up2, currentDepth, bias, pcfRadius);
    }
}

void main() {
    mat4 model       = v_model;
    uint objectIndex = v_objectIndex;

    // ---------- 圆角几何法线 ----------
    vec3  s        = halfSize;
    float rMax     = min(min(s.x, s.y), s.z) * 0.999;
    float r        = clamp(radius, 0.0, rMax);
    vec3  coreHalf = s - vec3(r);

    vec3 core = clamp(v_position, -coreHalf, coreHalf);
    vec3 n    = v_position - core;
    if (length(n) < 1e-5) {
        vec3 t = abs(v_position) / s;
        if (t.x >= t.y && t.x >= t.z) n = vec3(sign(v_position.x), 0.0, 0.0);
        else if (t.y >= t.z)          n = vec3(0.0, sign(v_position.y), 0.0);
        else                          n = vec3(0.0, 0.0, sign(v_position.z));
    }
    n = normalize(n);

    vec3 baseColor = v_position.x > -0.5 * halfSize.x ? baseColor_white : baseColor_green;

    // ---------- 图案 UV / 切线基 ----------
    float eps = 1e-3;
    vec2  uv  = vec2(0.0);
    bool  hit = false;
    vec3  T = vec3(0.0), B = vec3(0.0);

    if (patternFace == 4 && v_local.z >  s.z - eps) {
        uv  = v_position.xy / coreHalf.xy * 0.5 + 0.5;
        hit = all(lessThanEqual(abs(v_position.xy), coreHalf.xy + vec2(eps)));
        T = vec3( 1, 0, 0); B = vec3(0, 1, 0);
    }
    else if (patternFace == 5 && v_local.z < -s.z + eps) {
        uv  = vec2(-v_position.x, v_position.y) / coreHalf.xy * 0.5 + 0.5;
        hit = all(lessThanEqual(abs(v_position.xy), coreHalf.xy + vec2(eps)));
        T = vec3(-1, 0, 0); B = vec3(0, 1, 0);
    }
    else if (patternFace == 0 && v_local.x >  s.x - eps) {
        uv  = vec2(-v_position.z, v_position.y) / coreHalf.zy * 0.5 + 0.5;
        hit = all(lessThanEqual(abs(vec2(v_position.z, v_position.y)), coreHalf.zy + vec2(eps)));
        T = vec3(0, 0, -1); B = vec3(0, 1, 0);
    }
    else if (patternFace == 1 && v_local.x < -s.x + eps) {
        uv  = vec2( v_position.z, v_position.y) / coreHalf.zy * 0.5 + 0.5;
        hit = all(lessThanEqual(abs(vec2(v_position.z, v_position.y)), coreHalf.zy + vec2(eps)));
        T = vec3(0, 0, 1); B = vec3(0, 1, 0);
    }
    else if (patternFace == 2 && v_local.y >  s.y - eps) {
        uv  = vec2( v_position.x, -v_position.z) / coreHalf.xz * 0.5 + 0.5;
        hit = all(lessThanEqual(abs(vec2(v_position.x, v_position.z)), coreHalf.xz + vec2(eps)));
        T = vec3(1, 0, 0); B = vec3(0, 0, -1);
    }
    else if (patternFace == 3 && v_local.y < -s.y + eps) {
        uv  = vec2( v_position.x,  v_position.z) / coreHalf.xz * 0.5 + 0.5;
        hit = all(lessThanEqual(abs(vec2(v_position.x, v_position.z)), coreHalf.xz + vec2(eps)));
        T = vec3(1, 0, 0); B = vec3(0, 0, 1);
    }

    // ---------- 图案混合 + 凹凸 ----------
    vec3 color = baseColor;
    if (hit) {
        vec4 pat = samplePattern(uv);
        color = mix(baseColor, pat.rgb, pat.a);

        float whiteness   = min(min(pat.r, pat.g), pat.b);
        float colorAmount = 1.0 - smoothstep(0.4, 0.8, whiteness);

        vec2 texel = 1.0 / vec2(patternSize());
        vec2 off   = texel * 2.0;

        float hL = heightAt(uv - vec2(off.x, 0.0));
        float hR = heightAt(uv + vec2(off.x, 0.0));
        float hD = heightAt(uv - vec2(0.0, off.y));
        float hU = heightAt(uv + vec2(0.0, off.y));

        vec2 grad = vec2(hR - hL, hU - hD) * 0.5;

        n = normalize(n - carveDepth * colorAmount * (grad.x * T + grad.y * B));
    }

    vec3 nWorld = normalize(mat3(model) * n);

    // ---------- Blinn-Phong ----------
    vec3  toLight = lightPos - v_world;
    float dist    = length(toLight);
    vec3  l       = toLight / max(dist, 1e-5);
    float diff    = max(dot(nWorld, l), 0.0);

    vec3  viewDir = normalize(camPos - v_world);
    vec3  halfV   = normalize(l + viewDir);
    float spec    = pow(max(dot(nWorld, halfV), 0.0), shininess);

    bool  hl  = (highlightId >= 0) && (int(objectIndex) == highlightId);
    float amb = ambient + (hl ? 1.0 : 0.0);

    // ---------- 拾取 ----------
    if (ivec2(gl_FragCoord.xy) == mousePixel) {
        uint depthQ  = uint(clamp(gl_FragCoord.z, 0.0, 1.0) * 65535.0);
        uint packedK = (depthQ << 16) | (objectIndex & 0xFFFFu);
        atomicMin(pickPacked, packedK);
    }

    // ---------- 阴影 ----------
    float shadow  = computeShadow(nWorld);
    float visibility = 1.0 - shadow;

    vec3 lit = color * (vec3(amb) + lightColor * diff * lightIntensity * visibility)
             + lightColor * spec * lightIntensity * 0.5 * visibility;

    fragColor = vec4(lit, 1.0);
}