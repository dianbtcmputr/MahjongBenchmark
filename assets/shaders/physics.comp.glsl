#version 430
layout(local_size_x = 64) in;
struct Body {
    vec4 pos_invMass;   // xyz=质心,        w=invMass (0 表示静态)
    vec4 quat;          // 四元数 (x,y,z,w)
    vec4 linVel_rest;   // xyz=线速度,       w=restitution
    vec4 angVel_fric;   // xyz=角速度,       w=friction
    vec4 halfSize;      // xyz=世界空间半尺寸
    vec4 invInertia;    // xyz=局部逆惯性对角元
};

layout(std430, binding = 2) buffer ModelBuffer   { mat4 models[]; };
layout(std430, binding = 5) buffer PhysicsBuffer { Body bodies[]; };

uniform int   nBodies;
uniform float dt;
uniform vec3  gravity;
uniform float groundY;
uniform float linDamping;
uniform float angDamping;
uniform float modelScale;

uniform vec2  tableHalfXZ;
uniform float time;

// ---------- 水 ----------
uniform float waterY;
uniform float poolFloorY;
uniform float waterLinDrag;
uniform float waterAngDrag;
uniform float buoyancyGain;

// ---------- 四元数 ----------
vec4 quatMul(vec4 a, vec4 b) {
    return vec4(
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z);
}

mat3 quatToMat(vec4 q) {
    float xx=q.x*q.x, yy=q.y*q.y, zz=q.z*q.z;
    float xy=q.x*q.y, xz=q.x*q.z, yz=q.y*q.z;
    float wx=q.w*q.x, wy=q.w*q.y, wz=q.w*q.z;
    return mat3(
        1.0-2.0*(yy+zz), 2.0*(xy+wz),   2.0*(xz-wy),
        2.0*(xy-wz),     1.0-2.0*(xx+zz), 2.0*(yz+wx),
        2.0*(xz+wy),     2.0*(yz-wx),   1.0-2.0*(xx+yy));
}

vec3 applyInvInertia(mat3 R, vec3 invI, vec3 v) {
    vec3 local = transpose(R) * v;
    local *= invI;
    return R * local;
}

const float GROUND_TOL = 0.008;

float groundHeightAt(vec3 wp) {
    if (abs(wp.x) <= tableHalfXZ.x && abs(wp.z) <= tableHalfXZ.y) {
        return groundY;
    }
    return poolFloorY;
}

// ===========================================================================
// 地面接触：迭代式求解
//
// 相比单次通过版本，本函数：
//   1. 把 8 个角点预先算好，避免在迭代中重复 mat3*vec3
//   2. Pass 1 迭代 NITER 次，每次从当前速度重算 vn，
//      只施加"消除残余接近速度"所需要的 delta 冲量
//      → 单角接触的 Δω 在迭代中被自然摊销，不再一帧塞 0.55 rad/s
//   3. 位置修正按完整穿透量抬升（不分成"面/角"两档），
//      因为这一步已经不会产生跳变
// ===========================================================================
void collideGround(inout vec3 pos, inout vec3 linVel, inout vec3 angVel,
                   mat3 R, vec3 halfSize, vec3 invI,
                   float invMass, float rest, float fric) {
    vec3  n  = vec3(0.0, 1.0, 0.0);
    float gY = groundHeightAt(pos);

    // ---- 预计算 8 个角点世界偏移 ----
    vec3 rW[8];
    vec3 lp[8];
    int idx = 0;
    for (int cx=-1; cx<=1; cx+=2)
    for (int cy=-1; cy<=1; cy+=2)
    for (int cz=-1; cz<=1; cz+=2) {
        vec3 l = vec3(cx, cy, cz) * halfSize;
        lp[idx] = l;
        rW[idx] = R * l;
        idx++;
    }

    // ---- 统计接触角点 ----
    bool  touching[8];
    float maxPen = 0.0;
    int   nTouching = 0;
    for (int k = 0; k < 8; k++) {
        float pen = gY - (pos.y + rW[k].y);
        touching[k] = (pen > 0.0);
        if (pen > maxPen) maxPen = pen;
        if (touching[k]) nTouching++;
    }
    if (nTouching == 0) return;
    if (nTouching == 3) nTouching = 4;

    // -----------------------------------------------------------------------
    // Pass 1：迭代法向冲量
    // -----------------------------------------------------------------------
    float totalJn = 0.0;
    const int NITER = 4;

    for (int iter = 0; iter < NITER; iter++) {
        for (int k = 0; k < 8; k++) {
            if (!touching[k]) continue;

            vec3  r  = rW[k];
            vec3  v  = linVel + cross(angVel, r);
            float vn = dot(v, n);
            if (vn >= 0.0) continue;

            // 恢复系数只在第一次迭代、明显高速撞击时启用
            float e = 0.0;
            if (iter == 0 && vn < -2.0) e = rest;

            vec3  rn = cross(r, n);
            float kk = invMass + dot(cross(applyInvInertia(R, invI, rn), r), n);
            float jn = -(1.0 + e) * vn / max(kk, 1e-8);

            // 每次迭代只施加消除"当前残余速度"的最小冲量，
            // 立即写回速度 → 后续角点 / 迭代看到新状态
            vec3 J = jn * n;
            linVel += J * invMass;
            angVel += applyInvInertia(R, invI, cross(r, J));
            totalJn += jn;
        }
    }

    // -----------------------------------------------------------------------
    // Pass 2：摩擦（单次通过足够，因为多角点摩擦是耗散型的）
    // -----------------------------------------------------------------------
    {
        float gravJn = (-gravity.y) * dt / max(invMass, 1e-8);
        float budget = fric * max(totalJn, gravJn) / float(nTouching);

        for (int k = 0; k < 8; k++) {
            if (!touching[k]) continue;

            vec3  r   = rW[k];
            vec3  v   = linVel + cross(angVel, r);
            vec3  vt  = v - dot(v, n) * n;
            float vtl = length(vt);
            if (vtl < 1e-4) continue;

            vec3  t   = vt / vtl;
            vec3  rt  = cross(r, t);
            float kt  = invMass + dot(cross(applyInvInertia(R, invI, rt), r), t);
            float jt  = clamp(-vtl / max(kt, 1e-8), -budget, budget);

            vec3 JT = jt * t;
            linVel += JT * invMass;
            angVel += applyInvInertia(R, invI, cross(r, JT));
        }
    }

    // -----------------------------------------------------------------------
    // 位置修正：完整抬升，一步到位
    // 迭代求解已经让单步 Δv 很小，所以整体抬升不再引起"起跳"。
    // -----------------------------------------------------------------------
    if (maxPen > 0.0) {
        pos.y += min(maxPen, 0.05);
    }

    // -----------------------------------------------------------------------
    // 接触阻尼：只保留最温和的版本
    // 振荡由子步 + 迭代解决，不再靠强阻尼硬压
    // -----------------------------------------------------------------------
    if (nTouching > 0) {
        const float REF_DT = 1.0 / 60.0;
        float s = dt / REF_DT;

        linVel   *= pow(0.97, s);
        angVel   *= pow(0.97, s);
        linVel.xz *= pow(0.90, s);
    }
}

// ---------------------------------------------------------------------------
// 水体：浮力 + 躺平趋势 + 阻力 + 湍流
// ---------------------------------------------------------------------------
void applyWater(inout vec3 pos, inout vec3 linVel, inout vec3 angVel,
                mat3 R, vec3 halfSize) {
    float worldHalfY = abs(R[0].y) * halfSize.x
                     + abs(R[1].y) * halfSize.y
                     + abs(R[2].y) * halfSize.z;

    float botY = pos.y - worldHalfY;
    if (botY >= waterY) return;

    float subm = clamp((waterY - botY) / (2.0 * worldHalfY), 0.0, 1.0);

    float g = -gravity.y;
    linVel.y += g * subm * 2.0 * buoyancyGain * dt;

    {
        vec3  up     = vec3(0.0, 1.0, 0.0);
        vec3  localX = R[0];
        vec3  target = (dot(localX, up) >= 0.0) ? up : -up;
        vec3  alignAxis = cross(localX, target);
        angVel += alignAxis * 40.0 * dt * subm;
    }

    const float REF_DT = 1.0 / 60.0;
    float linD = pow(waterLinDrag, dt / REF_DT);
    float angD = pow(waterAngDrag, dt / REF_DT);

    // 向上逃逸时削弱线性阻尼，避免"刚起跳就被水按住"
    if (linVel.y > 0.0) {
        linD = mix(linD, 1.0, 0.5);
    }
    linVel *= linD;
    angVel *= angD;

    // 近表面阻尼：只对下沉方向生效，且按 dt 缩放（与子步数无关）
    float nearSurface = 1.0 - clamp(abs(pos.y - waterY) / (2.0 * worldHalfY), 0.0, 1.0);
    if (nearSurface > 0.0 && linVel.y < 0.0) {
        linVel.y *= pow(1.0 - 0.35 * nearSurface, dt / REF_DT);
    }

    float t    = time;
    float seed = pos.x * 1.7 + pos.y * 2.3 + pos.z * 3.1;
    linVel.y += sin(t * 1.4 + seed * 0.9) * 0.6 * dt * subm;
    vec3 drift = vec3(
        sin(t * 0.5 + pos.z * 0.6 + seed * 0.3),
        0.0,
        sin(t * 0.7 + pos.x * 0.6 + seed * 0.5)
    );
    linVel += drift * 0.35 * dt * subm;
    vec3 turbAng = vec3(
        sin(t * 1.1 + seed * 2.1),
        sin(t * 0.8 + seed * 1.7),
        sin(t * 1.3 + seed * 2.5)
    );
    angVel += turbAng * 1.8 * dt * subm;
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= nBodies) return;

    Body bi = bodies[i];
    vec3  pos      = bi.pos_invMass.xyz;
    float invMass  = bi.pos_invMass.w;
    vec4  quat     = bi.quat;
    vec3  linVel   = bi.linVel_rest.xyz;
    float rest     = bi.linVel_rest.w;
    vec3  angVel   = bi.angVel_fric.xyz;
    float fric     = bi.angVel_fric.w;
    vec3  halfSize = bi.halfSize.xyz;
    vec3  invI     = bi.invInertia.xyz;

    if (invMass <= 0.0) return;

    mat3 R = quatToMat(quat);

    // 1 外力
    linVel += gravity * dt;
    angVel *= angDamping;

    // 2 水
    applyWater(pos, linVel, angVel, R, halfSize);

    // 3 地面（迭代求解）
    collideGround(pos, linVel, angVel, R, halfSize, invI, invMass, rest, fric);

    // 4 牌对牌
    vec3 dLinAll = vec3(0.0);
    vec3 dAngAll = vec3(0.0);
    vec3 posCorr = vec3(0.0);
    bool anyContact = false;

    vec3 linVelSnap = linVel;
    vec3 angVelSnap = angVel;

    for (uint j = 0u; j < nBodies; ++j) {
        if (j == i) continue;

        Body bj = bodies[j];
        vec3  pj       = bj.pos_invMass.xyz;
        float invMassJ = bj.pos_invMass.w;
        mat3  Rj       = quatToMat(bj.quat);
        vec3  hsJ      = bj.halfSize.xyz;

        vec3  d     = pj - pos;
        float reach = length(halfSize) + length(hsJ);
        if (dot(d, d) > reach * reach) continue;

        vec3 axes[15];
        axes[0] = R[0];  axes[1] = R[1];  axes[2] = R[2];
        axes[3] = Rj[0]; axes[4] = Rj[1]; axes[5] = Rj[2];
        int nAx = 6;
        for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            axes[nAx++] = cross(R[a], Rj[b]);

        float minPen  = 1e9;
        vec3  minAxis = vec3(0.0);
        bool  sep     = false;
        for (int k = 0; k < 15; ++k) {
            vec3  L  = axes[k];
            float l2 = dot(L, L);
            if (l2 < 1e-8) continue;
            L *= inversesqrt(l2);

            float rA = abs(dot(R[0],  L)) * halfSize.x
                     + abs(dot(R[1],  L)) * halfSize.y
                     + abs(dot(R[2],  L)) * halfSize.z;
            float rB = abs(dot(Rj[0], L)) * hsJ.x
                     + abs(dot(Rj[1], L)) * hsJ.y
                     + abs(dot(Rj[2], L)) * hsJ.z;
            float dL      = dot(d, L);
            float overlap = rA + rB - abs(dL);
            if (overlap <= 0.0) { sep = true; break; }
            if (overlap < minPen) {
                minPen  = overlap;
                minAxis = (dL < 0.0) ? -L : L;
            }
        }
        if (sep) continue;

        float dA = dot(pos, minAxis);
        float dB = dot(pj,  minAxis);
        float dC = (dA + dB) * 0.5;
        vec3 contact = (pos + pj) * 0.5;
        contact += minAxis * (dC - dot(contact, minAxis));

        vec3 rI = contact - pos;
        vec3 rJ = contact - pj;

        vec3 n = -minAxis;

        {
            const float PEN_THRESHOLD = 0.005;
            float penCorr = minPen - PEN_THRESHOLD;
            if (penCorr > 0.0 && invMass + invMassJ > 0.0)
                posCorr += n * penCorr * 0.4
                         * (invMass / (invMass + invMassJ));
        }

        vec3 vAi = linVelSnap + cross(angVelSnap, rI);
        vec3 vBj = bj.linVel_rest.xyz + cross(bj.angVel_fric.xyz, rJ);
        vec3 vRel = vAi - vBj;
        float vn  = dot(vRel, n);
        if (vn > 0.0) continue;

        anyContact = true;

        vec3 rnI = cross(rI, n);
        vec3 rnJ = cross(rJ, n);
        float k = invMass + invMassJ
                + dot(cross(applyInvInertia(R,  invI, rnI), rI), n)
                + dot(cross(applyInvInertia(Rj, bj.invInertia.xyz, rnJ), rJ), n);
        float e  = min(rest, bj.linVel_rest.w);
        float jn = -(1.0 + e) * vn / max(k, 1e-8);

        vec3 J = jn * n;
        dLinAll += J * invMass;
        dAngAll += applyInvInertia(R, invI, cross(rI, J));

        vec3  vt  = vRel - vn * n;
        float vtl = length(vt);
        if (vtl > 1e-4) {
            vec3  t   = vt / vtl;
            vec3  rtI = cross(rI, t);
            vec3  rtJ = cross(rJ, t);

            float kt = invMass + invMassJ
                     + dot(cross(applyInvInertia(R,  invI, rtI), rI), t)
                     + dot(cross(applyInvInertia(Rj, bj.invInertia.xyz, rtJ), rJ), t);

            float jt    = -vtl / max(kt, 1e-8);
            float jtMax = fric * abs(jn);
            if (vtl < 0.02) jtMax = max(jtMax, abs(jt));
            jt = clamp(jt, -jtMax, jtMax);

            vec3 JT = jt * t;
            dLinAll += JT * invMass;
            dAngAll += applyInvInertia(R, invI, cross(rI, JT));
        }
    }

    linVel += dLinAll;
    angVel += dAngAll;

    const float MAX_ANG_PER_FRAME = 0.2;
    float maxAngVel = MAX_ANG_PER_FRAME / max(dt, 1e-5);
    float aw = length(angVel);
    if (aw > maxAngVel) angVel *= maxAngVel / aw;

    const float MAX_LIN = 50.0;
    float lv = length(linVel);
    if (lv > MAX_LIN) linVel *= MAX_LIN / lv;

    if (anyContact) {
        float v2 = dot(linVel, linVel);
        float w2 = dot(angVel, angVel);
        if (v2 < 1.0 && w2 < 25.0) {
            linVel *= 0.94;
            angVel *= 0.90;
        }
    }

    // 位置积分
    pos += posCorr;
    pos += linVel * dt;

    // 低速截断：完全脱离水面时才允许彻底冻结
    float whY = abs(R[0].y) * halfSize.x
              + abs(R[1].y) * halfSize.y
              + abs(R[2].y) * halfSize.z;
    bool fullyInAir = (pos.y - whY) > waterY;
    if (fullyInAir && dot(linVel, linVel) < 4e-4 && dot(angVel, angVel) < 2.5e-3) {
        linVel = vec3(0.0);
        angVel = vec3(0.0);
    }

    // 四元数积分
    vec4 omegaQ = vec4(angVel, 0.0);
    quat = normalize(quat + 0.5 * dt * quatMul(omegaQ, quat));

    // 写回物理状态
    bodies[i].pos_invMass.xyz = pos;
    bodies[i].quat            = quat;
    bodies[i].linVel_rest.xyz = linVel;
    bodies[i].angVel_fric.xyz = angVel;

    // 重建模型矩阵
    mat3 Rn = quatToMat(quat);
    mat4 m  = mat4(1.0);
    m[0] = vec4(Rn[0] * modelScale, 0.0);
    m[1] = vec4(Rn[1] * modelScale, 0.0);
    m[2] = vec4(Rn[2] * modelScale, 0.0);
    m[3] = vec4(pos, 1.0);
    models[i] = m;
}