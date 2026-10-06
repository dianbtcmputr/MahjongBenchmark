#version 430
#define SKY_ROTATE_SPEED %SKY_ROTATE_SPEED%

in vec3 v_world;
out vec4 fragColor;

// ---------- 相机 / 时间 ----------
uniform vec3  camPos;
uniform float time;

// ---------- 水的参数 ----------
uniform vec3  waterColor;     // 深水基调色
uniform float absorb;         // 吸收系数（1/m）

// ---------- 噪声 ----------
uniform float bumpStrength;   // 法线强度
uniform float refractStrength;
uniform float noiseScale;     // XY 频率
uniform float noiseSpeed;     // 时间推进速度
uniform float warpAmount;     // 域扭曲强度

// ---------- 池底 ----------
uniform float poolFloorY;
uniform float tileSize;
uniform float poolHalfSize;

// ---------- 光照 ----------
uniform vec3  lightPos;
uniform vec3  lightColor;
uniform float lightIntensity;
uniform float ambient;
uniform float shininess;

layout(binding = 3) uniform sampler3D noiseTex;
layout(binding = 4) uniform samplerCube skyTex;

// 3D 噪声（值噪声，纹理里存 0..1，映射到 -1..1）
float noiseAt(vec3 p) {
    return texture(noiseTex, p).r * 2.0 - 1.0;
}

float fbm(vec3 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 2; ++i) {
        v += a * noiseAt(p);
        p = p * 2.03 + vec3(1.7, 9.2, 3.5);   // 非整数倍 + 偏移，避免对齐
        a *= 0.5;
    }
    return v;
}

float heightAt(vec2 xz, float t) {
    vec3 p = vec3(xz * noiseScale, t * noiseSpeed);
    // 域扭曲：用两层 fbm 扰动 XY
    float wx = fbm(p + vec3(0.0, 0.0, 13.0));
    float wz = fbm(p + vec3(0.0, 0.0, 71.0));
    vec3 q  = vec3(p.xy + warpAmount * vec2(wx, wz), p.z);
    return fbm(q);
}

// 法线：从高度场梯度来
vec3 computeNormal(vec2 xz, float t) {
    // 让噪声空间里的差分步长 ≈ 0.2 个格点（G=16 → 0.0125 纹理单位）
    float e = 0.0125 / max(noiseScale, 1e-4);
    float hL = heightAt(xz - vec2(e, 0.0), t);
    float hR = heightAt(xz + vec2(e, 0.0), t);
    float hD = heightAt(xz - vec2(0.0, e), t);
    float hU = heightAt(xz + vec2(0.0, e), t);
    float dx = (hR - hL) / (2.0 * e);
    float dz = (hU - hD) / (2.0 * e);
    return normalize(vec3(-dx * bumpStrength, 1.0, -dz * bumpStrength));
}

mat3 rotationY(float angle) {
    float c = cos(angle);
    float s = sin(angle);

    return mat3(
        c,   0.0, -s,
        0.0, 1.0,  0.0,
        s,   0.0,  c
    );
}

// 抗锯齿棋盘格
float checkersGrad(vec2 p) {
    vec2 dpdx = dFdx(p);
    vec2 dpdy = dFdy(p);
    vec2 w    = abs(dpdx) + abs(dpdy) + 0.001;
    vec2 i    = 2.0 * (abs(fract((p - 0.5 * w) * 0.5) - 0.5)
                     - abs(fract((p + 0.5 * w) * 0.5) - 0.5)) / w;
    return 0.5 - 0.5 * i.x * i.y;
}

vec3 poolFloorColor(vec2 xz) {
    float c = checkersGrad(xz / tileSize);
    return mix(vec3(0.08), vec3(0.92), c);
}

// 天空
vec3 skyColor(vec3 dir) {
    return texture(skyTex, normalize(dir * rotationY(time * SKY_ROTATE_SPEED))).rgb * max(lightIntensity*0.8, 0.7);
}

void main() {
    vec2 xz = v_world.xz;

    // ---------- 法线 / 视线 ----------
    vec3 N = computeNormal(xz, time);

    // 光照/反射用完整法线
    vec3 V = normalize(camPos - v_world);

    // 折射用"压扁"的法线
    vec3 Nrefr = normalize(vec3(N.xz * refractStrength, N.y));

    const float ETA = 1.0 / 1.333;
    vec3 T = refract(-V, Nrefr, ETA);

    vec3  refr;
    float waterDepth;

    if (length(T) < 1e-6 || T.y >= -1e-4) {
        refr = waterColor;
        waterDepth = 0.0;
    } else {
        // 射线 P + t*T 与 y = poolFloorY 相交
        float tHit   = (poolFloorY - v_world.y) / T.y;
        vec3  Pfloor = v_world + tHit * T;

        refr = poolFloorColor(Pfloor.xz);

        waterDepth = max(v_world.y - poolFloorY, 0.0);
    }

    // 深度吸收：越深越偏向水色
    float absorbT = exp(-absorb * waterDepth);
    refr = mix(waterColor, refr, absorbT);

    // ---------- 反射----------
    vec3 R    = reflect(-V, N);
    vec3 refl = skyColor(R);

    // ---------- 菲涅尔 ----------
    float cosT = max(dot(N, V), 0.0);
    float F    = 0.02 + 0.98 * pow(1.0 - cosT, 4.0);

    // ---------- 点光源----------
    vec3  toLight = lightPos - v_world;
    float dist    = length(toLight);
    vec3  L       = toLight / max(dist, 1e-5);
    float diff    = max(dot(N, L), 0.0);

    vec3  H    = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), shininess);

    vec3 base  = mix(refr, refl, F);
    vec3 color = base * (vec3(ambient) + lightColor * diff * lightIntensity)
               + lightColor * spec * lightIntensity * 0.5;

    fragColor = vec4(mix(color, refl, F*F), 1.0);
    
    float dist2center = length(v_world.xz);
    fragColor.a *= 1 - 4 * (1 / poolHalfSize) * max(dist2center - (poolHalfSize * 0.75), 0);
}