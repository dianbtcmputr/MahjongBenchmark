from time import perf_counter
from random import uniform, randint
from os import listdir
from glutils import *

TARGET_FPS = SETTINGS.target_fps

N_MAHJONG  = SETTINGS.n_mahjong
CAPACITY   = N_MAHJONG + 1
TABLE_SLOT = N_MAHJONG

SHADOW_SIZE = 4096
SHADOW_FAR  = 100.0

WATER_Y        = -1.2
POOL_FLOOR_Y   = -4.0
POOL_HALF_SIZE = 100.0
TILE_SIZE      = 2.0

PHYSICS_DT      = 1.0 / 60.0
MAX_SUBSTEPS    = 5 # 单帧最多补 5 步，防止死亡螺旋
MAX_FRAME_TIME  = 0.25
TIMESCALE = 1.0
N_SUB = SETTINGS.physics_sub # 每物理步拆成若干个子步

DAY_SECONDS   = 15.0
NIGHT_SECONDS = 10.0
CYCLE_SECONDS = DAY_SECONDS + NIGHT_SECONDS

SUN_RADIUS  = 30.0
SUN_TILT_X  = -10.0

_WARM_COLOR  = glm.vec3(1.00, 0.45, 0.15)   # 日出 / 日落
_NOON_COLOR  = glm.vec3(1.00, 1.00, 1.00)   # 正午
_NIGHT_COLOR = glm.vec3(0.20, 0.30, 0.60)   # 深夜

_NIGHT_INT = 0.3 # 夜晚光源最低强度
_NIGHT_SKY = 0.3 # 夜晚天空最低亮度

shadow_mode = SETTINGS.shadow_mode
pcf_radius  = 0.005
shadow_bias = 0.0001

cam = Camera()
cam.distance = 25

def _cycle_t(now: float) -> float:
    """真实时间 → [0,1) 昼夜相位"""
    e = (now - T0) % CYCLE_SECONDS
    if e < DAY_SECONDS:
        return 0.5 * (e / DAY_SECONDS)              # 白天：0 → 0.5
    return 0.5 + 0.5 * ((e - DAY_SECONDS) / NIGHT_SECONDS)   # 夜晚：0.5 → 1

def compute_sky(now: float):
    t   = _cycle_t(now)
    phi = t * 6.2831
    h   = glm.sin(phi)
    sun = glm.vec3(
        SUN_TILT_X,
        SUN_RADIUS * h,
        SUN_RADIUS * glm.cos(phi),
    )

    soft_y = glm.sqrt(sun.y * sun.y + 4.0) * 0.5 + 0.5
    light_pos = glm.vec3(sun.x, soft_y, sun.z)

    noon_w  = glm.smoothstep(0.0, 0.40, h)     # 0→1 随太阳升高
    night_w = glm.smoothstep(0.0, -0.25, h)    # 0→1 随太阳沉入地平线以下

    day_w = glm.smoothstep(0.0, 0.50, h)

    day_int = 0.20 + 0.80 * day_w
    intensity = day_int + (_NIGHT_INT - day_int) * night_w

    day_sky = 0.20 + 0.80 * day_w
    sky_brightness = day_sky + (_NIGHT_SKY - day_sky) * night_w

    day_col = _WARM_COLOR + (_NOON_COLOR - _WARM_COLOR) * noon_w
    color   = day_col     + (_NIGHT_COLOR - day_col)   * night_w

    dusk_w = 1.0 - glm.smoothstep(0.0, 0.5, abs(h))

    return (light_pos, tuple(color), intensity, sky_brightness,
            dusk_w, tuple(sun))

init_pick_ssbo()
init_shadow_map(size=SHADOW_SIZE, far_plane=SHADOW_FAR)

store = InstanceStore(CAPACITY)

MJ_FILES = ["tiles\\zhong.png", "tiles\\fa.png", "tiles\\bai.png"] # listdir(join(IMAGE_PATH, "tiles"))
mj_tex_array, N_LAYERS = create_texture_array(MJ_FILES)

table_tex = load_texture("table.png")

sky_tex = load_cube_map("skybox")
sky_vbo, sky_nverts = create_skybox_cube()

water_vbo, water_nverts = create_water_quad(POOL_HALF_SIZE, WATER_Y)
water_tex = WaterNoise3D(glm.ivec3(32)).id # seed 1337

glActiveTexture(GL_TEXTURE0)
glBindTexture(GL_TEXTURE_2D_ARRAY, mj_tex_array)
glActiveTexture(GL_TEXTURE1)
glBindTexture(GL_TEXTURE_2D, table_tex.id)
glActiveTexture(GL_TEXTURE2)
glBindTexture(GL_TEXTURE_CUBE_MAP, get_shadow_cube())
glActiveTexture(GL_TEXTURE3)
glBindTexture(GL_TEXTURE_3D, water_tex)
glActiveTexture(GL_TEXTURE4)
glBindTexture(GL_TEXTURE_CUBE_MAP, sky_tex)

prog_mahjong = load_shader_vf("mahjong", PCF_SAMPLES=SETTINGS.pcf_samples)
prog_shadow  = load_shader_vf("shadow")
prog_water   = load_shader_vf("water", SKY_ROTATE_SPEED=0.05)
prog_skybox  = load_shader_vf("skybox", SKY_ROTATE_SPEED=0.05)
prog_physics = load_shader_comp("physics")

mj_geom = MahjongGeometry(n_hint=SETTINGS.round_n_hint)
tb_geom = TableGeometry(n_hint=2 * SETTINGS.round_n_hint)

mahjongs = []
for x in range(N_MAHJONG):
    cfg = {
        "slot":      x,
        "object_id": x + 1,
        "position":  glm.vec3(uniform(-9, 9), uniform(1, 6), uniform(-9, 9)),
        "rot_axis":  glm.vec3(uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)),
        "angle_deg": uniform(-90, 90),
        "scale":     1.0,
        "tex_layer": randint(0, N_LAYERS - 1),
    }
    mahjongs.append(cfg)
    store.set_transform(cfg["slot"], cfg["position"], cfg["angle_deg"],
                        cfg["rot_axis"], cfg["scale"], upload=False)
    store.set_meta(cfg["slot"], cfg["object_id"], cfg["tex_layer"], upload=False)

store.set_transform(
    TABLE_SLOT, position=(0, -1, 0),
    angle_deg=0, axis=(0, 1, 0), scale=1.0, upload=False)
store.set_meta(TABLE_SLOT, object_id=0, tex_layer=0, upload=False)
store.upload_all()

init_physics_buffers(N_MAHJONG)

for i, cfg in enumerate(mahjongs):
    q = quat_from_axis_angle(
        cfg["rot_axis"],
        glm.radians(cfg["angle_deg"])
    )
    s = cfg["scale"]
    init_physics_body(
        i,
        tuple(cfg["position"]),
        q,
        (0.25 * s, 0.75 * s, 0.5 * s),   # 世界空间半尺寸 = 几何 halfs × scale
        mass=1.0,
        restitution=0.05,
        friction=0.3,
    )

GEOM_LIST = [
    (mj_geom, N_MAHJONG, 0,          mj_geom.halfs, mj_geom.radius),
    (tb_geom, 1,         TABLE_SLOT, tb_geom.halfs, tb_geom.radius),
]

T0 = perf_counter()

last = last_hud = perf_counter()
last_hit    = -1
mouse_px    = glm.ivec2(0)
accumulator = 0.0
info = ""

def init_const_uniforms():
    prog_skybox["water_y"] = WATER_Y
    prog_skybox["poolHalfSize"] = POOL_HALF_SIZE

    prog_mahjong["carveDepth"]  = 32.0
    prog_mahjong["shadowMode"]  = shadow_mode
    prog_mahjong["pcfRadius"]   = pcf_radius
    prog_mahjong["shadowBias"]  = shadow_bias
    prog_mahjong["shadowFar"]   = SHADOW_FAR

    prog_water["waterColor"]   = (0.10, 0.60, 0.60)
    prog_water["absorb"]       = 0.06
    prog_water["bumpStrength"] = 4.0
    prog_water["refractStrength"] = 0.06
    prog_water["noiseScale"]   = 0.01
    prog_water["noiseSpeed"]   = 0.03
    prog_water["warpAmount"]   = 0.4
    prog_water["poolFloorY"]   = POOL_FLOOR_Y
    prog_water["poolHalfSize"] = POOL_HALF_SIZE
    prog_water["tileSize"]     = TILE_SIZE
    prog_water["shininess"]      = 128.0

@win.event
def on_draw():
    global last, last_hit, accumulator, last_hud, info

    # ---------- 帧时长 ----------
    now = perf_counter()
    frame_time = now - last
    last = now

    light_pos, light_color, light_intensity, sky_brightness, dusk_w, sun_pos = compute_sky(now)

    if frame_time > MAX_FRAME_TIME:
        frame_time = MAX_FRAME_TIME

    accumulator += frame_time * TIMESCALE

    # ---------- 物理 ----------
    SUB_DT = PHYSICS_DT / N_SUB

    substeps = 0
    while accumulator >= PHYSICS_DT and substeps < MAX_SUBSTEPS:
        for _ in range(N_SUB):
            step_physics(
                prog_physics, store.mat_ssbo,
                N_MAHJONG, SUB_DT,              # ← 用子步 dt
                gravity=(0.0, -49, 0.0),
                ground_y=-0.5,
                model_scale=1.0,
                table_half_xz=(10.0, 10.0),
                water_y=WATER_Y,
                pool_floor_y=POOL_FLOOR_Y,
                water_lin_drag=0.9,
                water_ang_drag=0.85,
                buoyancy_gain=1.0,
                time=now - T0
            )
        accumulator -= PHYSICS_DT
        substeps += 1

    if substeps >= MAX_SUBSTEPS:
        accumulator = 0.0

    # ---------- 阴影 ----------
    if shadow_mode != SHADOW_NONE:
        render_shadow_cube(
            prog_shadow, light_pos, GEOM_LIST,
            far_plane=SHADOW_FAR
        )

    # ---------- 主渲染 ----------
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    cam_proj = cam.proj_matrix
    cam_view = cam.view_matrix
    cam_pos = cam.position

    # ---------- 天空盒 ----------
    prog_skybox.use()
    prog_skybox["skyBrightness"] = sky_brightness
    prog_skybox["duskWeight"] = dusk_w
    prog_skybox["duskColor"]  = (1.00, 0.55, 0.20)
    prog_skybox["sunPos"] = sun_pos
    prog_skybox["time"] = now - T0

    uniform_matrix(prog_skybox, "proj", cam_proj)
    uniform_matrix(prog_skybox, "view", cam_view)

    glBindBuffer(GL_ARRAY_BUFFER, sky_vbo)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), None)
    glEnableVertexAttribArray(0)
    glDrawArrays(GL_TRIANGLES, 0, sky_nverts)

    prog_mahjong.use()
    clear_pick_ssbo()

    prog_mahjong["mousePixel"]  = tuple(mouse_px)
    prog_mahjong["camPos"]      = tuple(cam_pos)
    prog_mahjong["lightPos"]    = tuple(light_pos)
    prog_mahjong["lightColor"]  = light_color

    uniform_matrix(prog_mahjong, "proj", cam_proj)
    uniform_matrix(prog_mahjong, "view", cam_view)

    # ---- 麻将批 ----
    prog_mahjong["useTableTex"]     = 0
    prog_mahjong["halfSize"]        = mj_geom.halfs
    prog_mahjong["radius"]          = mj_geom.radius
    prog_mahjong["patternFace"]     = mj_geom.tex_face
    prog_mahjong["baseColor_white"] = (1.0, 1.0, 1.0)
    prog_mahjong["baseColor_green"] = (0.0, 0.5, 1.0)
    prog_mahjong["lightIntensity"]  = 0.7 * light_intensity
    prog_mahjong["ambient"]         = 0.3 * sky_brightness
    prog_mahjong["shininess"]       = 64.0
    prog_mahjong["highlightId"]     = last_hit if last_hit > 0 else -1
    prog_mahjong["instanceOffset"]  = 0
    draw_instanced(mj_geom, N_MAHJONG)

    # ---- 桌子 ----
    prog_mahjong["useTableTex"]     = 1
    prog_mahjong["halfSize"]        = tb_geom.halfs
    prog_mahjong["radius"]          = tb_geom.radius
    prog_mahjong["patternFace"]     = tb_geom.tex_face
    prog_mahjong["baseColor_white"] = (107/255, 60/255, 19/255)
    prog_mahjong["baseColor_green"] = (107/255, 60/255, 19/255)
    prog_mahjong["lightIntensity"]  = 0.3 * light_intensity
    prog_mahjong["ambient"]         = 0.4 * sky_brightness
    prog_mahjong["shininess"]       = 1.0
    prog_mahjong["highlightId"]     = -1
    prog_mahjong["instanceOffset"]  = TABLE_SLOT
    draw_instanced(tb_geom, 1)

    # ---------- 水面 ----------
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glDepthMask(GL_FALSE)

    prog_water.use()
    uniform_matrix(prog_water, "proj", cam_proj)
    uniform_matrix(prog_water, "view", cam_view)

    prog_water["camPos"]       = tuple(cam_pos)
    prog_water["time"]         = now - T0

    prog_water["lightPos"]       = tuple(light_pos)
    prog_water["lightColor"]     = light_color
    prog_water["lightIntensity"] = 1.5 * light_intensity
    prog_water["ambient"]        = 0.5 * sky_brightness

    draw_water(water_vbo, water_nverts)

    glDepthMask(GL_TRUE)
    glDisable(GL_BLEND)

    hit = read_pick_ssbo()
    if hit:
        last_hit = hit[0]
    else:
        last_hit = -1

    # ---------- HUD ----------
    if now - last_hud > 0.5:
        last_hud = now
        fps = round(1.0 / max(frame_time, 1e-5), 0)
        info = f"FPS: {fps}"

    Label(
        info, font_size=win.height / 25,
        x=win.height * 0.02,
        y=win.height * 0.98,
        anchor_x="left",
        anchor_y="top").draw()

@win.event
def on_mouse_motion(x, y, dx, dy):
    mouse_px.x = x
    mouse_px.y = y
    cam.apply_mouse_motion(x, y, win)

@win.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    on_mouse_motion(x, y, dx, dy)

@win.event
def on_key_press(symbol, modifiers):
    global shadow_mode, pcf_radius, shadow_bias
    if symbol == key.SPACE:
        for slot in range(N_MAHJONG):
            lin = (
                uniform(-10.0, 10.0),    # 水平随机
                uniform(20.0, 50.0),    # 向上抛
                uniform(-10.0, 10.0),
            )
            ang = (
                uniform(-12.0, 12.0),  # 翻滚
                uniform(-12.0, 12.0),
                uniform(-12.0, 12.0),
            )
            set_body_velocity(slot, lin_vel=lin, ang_vel=ang)

@win.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    if scroll_y < 0:
        cam.distance *= 1.1
        if cam.distance > POOL_HALF_SIZE:
            cam.distance /= 1.1
    else:
        cam.distance /= 1.1
        if cam.distance < 5:
            cam.distance *= 1.1

PUSH_SPEED   = 30.0
PUSH_UPWARDS = 10.0

@win.event
def on_mouse_press(x, y, button, modifiers):
    hit = read_pick_ssbo()
    if not hit:
        return
    oid = hit[0]
    if oid == 0:
        return # 0 是桌子
    slot = oid - 1
    if not (0 <= slot < N_MAHJONG):
        return

    pos, _, _, _ = get_body_state(slot)
    cam_pos = cam.position

    dx = pos.x - cam_pos.x
    dz = pos.z - cam_pos.z
    l2 = dx * dx + dz * dz
    if l2 < 1e-6:
        return # 牌恰好在相机正上/正下方

    inv = 1.0 / (l2 ** 0.5)
    dirx, dirz = dx * inv, dz * inv

    lin = (
        dirx * PUSH_SPEED,
        PUSH_UPWARDS,
        dirz * PUSH_SPEED,
    )

    ang = (
        uniform(-12.0, 12.0),
        uniform(-12.0, 12.0),
        uniform(-12.0, 12.0),
    )

    set_body_velocity(slot, lin_vel=lin, ang_vel=ang)

print("麻将数:", N_MAHJONG, " 麻将图案层数:", N_LAYERS)

init_const_uniforms()
app.run(1 / TARGET_FPS)
