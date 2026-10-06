from ctypes import create_string_buffer, sizeof
from noise3d import WaterNoise3D
from os.path import join
from PIL import Image as PILImage
from pyglet import app
from pyglet.gl import *
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.image import Texture, load as load_image
from pyglet.text import Label
from pyglet.window import key
from pyglet.window import Window
from pyglm import glm
from settings import SETTINGS

SHADER_PATH = "assets\\shaders"
IMAGE_PATH = "assets\\images"

GL_TEXTURE_MAX_ANISOTROPY_EXT = 0x84FE
GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT = 0x84FF

if SETTINGS.enable_msaa:
    _config = Config(
        major_version=4,
        minor_version=3,
        double_buffer=True,
        sample_buffers=1,
        samples=4,
    )
else:
    _config = Config(
        major_version=4,
        minor_version=3,
        double_buffer=True,
        depth_size=24,
    )

win = Window(resizable=True, vsync=SETTINGS.enable_vsync, config=_config)

SUPPORT_ANISOTROPIC = gl_info.have_extension('GL_EXT_texture_filter_anisotropic')
if SUPPORT_ANISOTROPIC:
    _max_aniso = (GLfloat * 1)()
    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, _max_aniso)
    ANISO_LEVEL: float = _max_aniso[0]

glEnable(GL_DEPTH_TEST)
glDepthFunc(GL_LEQUAL)

if SETTINGS.enable_msaa:
    glEnable(GL_MULTISAMPLE)

_vaos = (GLuint * 1)()
glGenVertexArrays(1, _vaos)
vao: int = _vaos[0]
glBindVertexArray(vao)

def _shader(filename: str, typ: str, **defines):
    with open(join(SHADER_PATH, filename), encoding="utf-8") as f:
        code = f.read()
        for k, v in defines.items():
            code = code.replace(f"%{k}%", str(v))
        return Shader(code, typ)

def load_shader_vf(name: str, **defines):
    return ShaderProgram(
        _shader(f"{name}.vert.glsl", "vertex", **defines),
        _shader(f"{name}.frag.glsl", "fragment", **defines)
    )

def load_shader_comp(name: str, **defines):
    with open(join(SHADER_PATH, f"{name}.comp.glsl"), encoding="utf-8") as f:
        code = f.read()
        for k, v in defines.items():
            code = code.replace(f"%{k}%", str(v))
    return ShaderProgram(Shader(code, "compute"))

def get_uniloc(program: ShaderProgram, name: str):
    return glGetUniformLocation(
        program.id,
        create_string_buffer(name.encode("utf-8"))
    )

def generate_subdivided_box(halfs: tuple[float, float, float], n_hint=64) -> list[float]:
    sx, sy, sz = halfs
    verts = []
    half_arr = [sx, sy, sz]
    base_step = max(halfs) * 2.0 / n_hint

    face_defs = [
        (0, 1, 2, sy, sz),   # ±X : ê_y × ê_z = +x̂
        (1, 2, 0, sz, sx),   # ±Y : ê_z × ê_x = +ŷ
        (2, 0, 1, sx, sy),   # ±Z : ê_x × ê_y = +ẑ
    ]

    for axis, u_ax, v_ax, uh, vh in face_defs:
        nu = max(1, int(round(2 * uh / base_step)))
        nv = max(1, int(round(2 * vh / base_step)))

        for sign_n in (+1.0, -1.0):
            for i in range(nu):
                for j in range(nv):
                    u0 = -uh + 2 * uh * i / nu
                    u1 = -uh + 2 * uh * (i + 1) / nu
                    v0 = -vh + 2 * vh * j / nv
                    v1 = -vh + 2 * vh * (j + 1) / nv

                    def pos(u, v):
                        p = [0.0, 0.0, 0.0]
                        p[axis] = sign_n * half_arr[axis]
                        p[u_ax] = u
                        p[v_ax] = v
                        return p

                    p00 = pos(u0, v0); p10 = pos(u1, v0)
                    p11 = pos(u1, v1); p01 = pos(u0, v1)

                    if sign_n > 0:
                        verts.extend(p00); verts.extend(p10); verts.extend(p11)
                        verts.extend(p00); verts.extend(p11); verts.extend(p01)
                    else:
                        verts.extend(p00); verts.extend(p01); verts.extend(p11)
                        verts.extend(p00); verts.extend(p11); verts.extend(p10)
    return verts

def create_float_vbo(data_l: list[float], existing_vbo: int = -1) -> int:
    if existing_vbo == -1:
        vbos = (GLuint * 1)()
        glGenBuffers(1, vbos)
        vbo = vbos[0]
    else:
        vbo = existing_vbo

    data = (GLfloat * len(data_l))(*data_l)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)

    return vbo

def create_texture_array(filenames: list[str],
                         mipmap: bool = True,
                         anisotropic: bool = True) -> tuple[int, int]:
    """从 IMAGE_PATH 下的一组文件创建 2D 数组纹理。
    要求所有图等尺寸。返回 (tex_id, layer_count)。"""
    images = [load_image(join(IMAGE_PATH, fn)) for fn in filenames]
    if not images:
        raise ValueError("空文件列表")
    w, h = images[0].width, images[0].height
    for img in images:
        assert img.width == w and img.height == h, \
            f"纹理数组要求所有图等尺寸，遇到 {img.width}x{img.height}"
    n = len(images)

    texs = (GLuint * 1)()
    glGenTextures(1, texs)
    tex_id = texs[0]

    glBindTexture(GL_TEXTURE_2D_ARRAY, tex_id)
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8,
                 w, h, n, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, None)

    for i, img in enumerate(images):
        data = img.get_data("RGBA", img.width * 4)
        glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0,
                        0, 0, i,
                        w, h, 1,
                        GL_RGBA, GL_UNSIGNED_BYTE, data)

    if mipmap:
        glGenerateMipmap(GL_TEXTURE_2D_ARRAY)

    if anisotropic and SUPPORT_ANISOTROPIC:
        glTexParameterf(GL_TEXTURE_2D_ARRAY,
                        GL_TEXTURE_MAX_ANISOTROPY_EXT, ANISO_LEVEL)

    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    if mipmap:
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER,
                        GL_LINEAR_MIPMAP_LINEAR)
    else:
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    return tex_id, n

def create_skybox_cube() -> tuple[int, int]:
    v = [
        # +X
         1,-1,-1,  1, 1,-1,  1, 1, 1,
         1,-1,-1,  1, 1, 1,  1,-1, 1,
        # -X
        -1,-1, 1, -1, 1, 1, -1, 1,-1,
        -1,-1, 1, -1, 1,-1, -1,-1,-1,
        # +Y
        -1, 1,-1,  1, 1,-1,  1, 1, 1,
        -1, 1,-1,  1, 1, 1, -1, 1, 1,
        # -Y
        -1,-1, 1,  1,-1, 1,  1,-1,-1,
        -1,-1, 1,  1,-1,-1, -1,-1,-1,
        # +Z
        -1,-1, 1, -1, 1, 1,  1, 1, 1,
        -1,-1, 1,  1, 1, 1,  1,-1, 1,
        # -Z
         1,-1,-1,  1, 1,-1, -1, 1,-1,
         1,-1,-1, -1, 1,-1, -1,-1,-1,
    ]
    vbo = create_float_vbo(v)
    return vbo, 36

class BoxGeometry:
    def __init__(self, halfs, radius: float, tex_face: int = -1, n_hint: int = 64):
        self.halfs = tuple(halfs)
        self.radius = radius
        self.tex_face = tex_face

        vbos = (GLuint * 1)()
        glGenBuffers(1, vbos)
        self.vbo = vbos[0]

        data = generate_subdivided_box(self.halfs, n_hint)
        self.n_vertices = len(data) // 3
        create_float_vbo(data, self.vbo)

class MahjongGeometry(BoxGeometry):
    def __init__(self, n_hint: int = 64):
        super().__init__((0.25, 0.75, 0.5), 0.08, tex_face=0, n_hint=n_hint)

class TableGeometry(BoxGeometry):
    def __init__(self, n_hint: int = 128):
        super().__init__((10.0, 0.5, 10.0), 0.2, tex_face=2, n_hint=n_hint)

class InstanceStore:
    """管理实例化渲染的 SSBO。

    - `binding 2: mat4 models[]`
    - `binding 3: uint objectIds[]`
    - `binding 4: uint texLayers[]`
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._mat   = [glm.mat4(1.0) for _ in range(capacity)]
        self._id    = [0] * capacity
        self._layer = [0] * capacity

        ssbos = (GLuint * 3)()
        glGenBuffers(3, ssbos)
        self.mat_ssbo   = ssbos[0]
        self.id_ssbo    = ssbos[1]
        self.layer_ssbo = ssbos[2]

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.mat_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, capacity * 64, None, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self.mat_ssbo)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.id_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, capacity * 4, None, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self.id_ssbo)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.layer_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, capacity * 4, None, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, self.layer_ssbo)

    def _write_mat(self, slot: int):
        m = self._mat[slot]
        data = (GLfloat * 16)()
        for c in range(4):
            for r in range(4):
                data[c * 4 + r] = m[c][r]
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.mat_ssbo)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, slot * 64, 64, data)

    def _write_id(self, slot: int):
        data = (GLuint * 1)(self._id[slot])
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.id_ssbo)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, slot * 4, 4, data)

    def _write_layer(self, slot: int):
        data = (GLuint * 1)(self._layer[slot])
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.layer_ssbo)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, slot * 4, 4, data)

    def set_transform(self, slot: int,
                      position=(0.0, 0.0, 0.0),
                      angle_deg: float = 0.0,
                      axis=(0.0, 1.0, 0.0),
                      scale: float = 1.0,
                      upload: bool = True):
        """重算并写入某个槽位的模型矩阵"""
        m = glm.translate(glm.mat4(1.0), glm.vec3(*position))
        m = glm.rotate(m, glm.radians(angle_deg), glm.vec3(*axis))
        m = glm.scale(m, glm.vec3(scale))
        self._mat[slot] = m
        if upload:
            self._write_mat(slot)

    def set_meta(self, slot: int, object_id: int, tex_layer: int,
                 upload: bool = True):
        """写入 object_id 和 tex_layer。"""
        self._id[slot]    = object_id
        self._layer[slot] = tex_layer
        if upload:
            self._write_id(slot)
            self._write_layer(slot)

    def set_layer(self, slot: int, tex_layer: int):
        """只换贴图层。"""
        self._layer[slot] = tex_layer
        self._write_layer(slot)

    def upload_all(self):
        n = self.capacity

        mat_data = (GLfloat * (n * 16))()
        for i, m in enumerate(self._mat):
            for c in range(4):
                for r in range(4):
                    mat_data[i * 16 + c * 4 + r] = m[c][r]
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.mat_ssbo)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * 64, mat_data)

        id_data = (GLuint * n)(*self._id)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.id_ssbo)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * 4, id_data)

        layer_data = (GLuint * n)(*self._layer)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.layer_ssbo)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * 4, layer_data)

def draw_instanced(geom: BoxGeometry, count: int):
    glBindBuffer(GL_ARRAY_BUFFER, geom.vbo)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                          3 * sizeof(GLfloat), None)
    glEnableVertexAttribArray(0)
    glDrawArraysInstanced(GL_TRIANGLES, 0, geom.n_vertices, count)

class Camera:
    def __init__(self):
        self.target = glm.vec3(0.0)
        self._y_up = glm.vec3(0.0, 1.0, 0.0)

        self.fov_rad = glm.radians(60.0)
        self.distance = 2.0
        self.alpha_rad = 0.0
        self.beta_rad = glm.radians(30.0)

        self._rest_alpha_rad = self.alpha_rad
        self._rest_beta_rad  = self.beta_rad
        self.max_alpha_dev = glm.radians(80.0)
        self.max_beta_dev  = glm.radians(60.0)

    @property
    def proj_matrix(self):
        return glm.perspective(self.fov_rad, win.aspect_ratio, 0.5, 400)

    @property
    def fov_deg(self):
        return glm.degrees(self.fov_rad)
    
    @fov_deg.setter
    def fov_deg(self, val: float):
        self.fov_rad = glm.radians(val)

    @property
    def alpha_deg(self):
        return glm.degrees(self.alpha_rad)
    
    @alpha_deg.setter
    def alpha_deg(self, val: float):
        self.alpha_rad = glm.radians(val)

    @property
    def beta_deg(self):
        return glm.degrees(self.beta_rad)
    
    @beta_deg.setter
    def beta_deg(self, val: float):
        self.beta_rad = glm.radians(val)

    @property
    def position(self):
        r = self.distance * glm.cos(self.beta_rad)
        y = self.distance * glm.sin(self.beta_rad)

        x = r * glm.cos(self.alpha_rad)
        z = r * glm.sin(self.alpha_rad)
        return glm.vec3(x, y, z)

    @property
    def view_matrix(self):
        return glm.lookAt(self.position, self.target, self._y_up)

    def apply_mouse_motion(self, x, y, win: Window):
        cx = win.width  * 0.5
        cy = win.height * 0.5

        nx = (x - cx) / max(cx, 1.0)
        ny = (y - cy) / max(cy, 1.0)
        nx = max(-1.0, min(1.0, nx))
        ny = max(-1.0, min(1.0, ny))

        sx = - glm.tanh(nx * 2.0)
        sy = glm.tanh(ny * 2.0) / 3

        self.alpha_rad = self._rest_alpha_rad + sx * self.max_alpha_dev
        self.beta_rad  = self._rest_beta_rad  + sy * self.max_beta_dev

        limit = glm.radians(89.0)
        self.beta_rad = glm.clamp(self.beta_rad, -limit, limit)

def load_texture(
        filename: str,
        mipmap: bool = SETTINGS.use_mipmap,
        anisotropic: bool = SETTINGS.use_anisotropy):
    img = load_image(join(IMAGE_PATH, filename))
    tex = img.get_texture()

    glBindTexture(GL_TEXTURE_2D, tex.id)

    if mipmap:
        glGenerateMipmap(GL_TEXTURE_2D)

    if anisotropic and SUPPORT_ANISOTROPIC:
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, ANISO_LEVEL)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    if mipmap:
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    else:
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    return tex

def _load_cube_map(
        filenames: list[str],
        mipmap: bool = True,
        anisotropic: bool = True) -> int:
    """filenames 顺序: [+X, -X, +Y, -Y, +Z, -Z]"""
    assert len(filenames) == 6, "天空盒必须是 6 张图"

    imgs = [PILImage.open(join(IMAGE_PATH, fn)).convert("RGBA")
            for fn in filenames]
    w, h = imgs[0].size
    for im in imgs:
        assert im.size == (w, h), \
            f"立方体面必须等尺寸，遇到 {im.size[0]}x{im.size[1]}"

    texs = (GLuint * 1)()
    glGenTextures(1, texs)
    tex_id = texs[0]

    glBindTexture(GL_TEXTURE_CUBE_MAP, tex_id)
    for i, im in enumerate(imgs):
        # 立方体贴图恰要求左上角为图像原点
        data = im.tobytes()
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0,
                     GL_RGBA8, w, h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, data)

    if mipmap:
        glGenerateMipmap(GL_TEXTURE_CUBE_MAP)

    if anisotropic and SUPPORT_ANISOTROPIC:
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, ANISO_LEVEL)

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER,
                    GL_LINEAR_MIPMAP_LINEAR if mipmap else GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    return tex_id

def load_cube_map(direc: str, suff: str = "png"):
    return _load_cube_map([
        join(direc, f"px.{suff}"),
        join(direc, f"nx.{suff}"),
        join(direc, f"py.{suff}"),
        join(direc, f"ny.{suff}"),
        join(direc, f"pz.{suff}"),
        join(direc, f"nz.{suff}"),
    ])

def uniform_matrix(program: ShaderProgram, name: str, matrix4: glm.mat4):
    loc = get_uniloc(program, name)
    glProgramUniformMatrix4fv(
        program.id, loc, 1, False,
        glm.value_ptr(matrix4)
    )

_PICK_EMPTY = 0xFFFFFFFF

pick_ssbo_id: int = -1

def init_pick_ssbo():
    global pick_ssbo_id
    ssbos = (GLuint * 1)()
    glGenBuffers(1, ssbos)
    pick_ssbo_id = ssbos[0]
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, pick_ssbo_id)
    glBufferData(GL_SHADER_STORAGE_BUFFER, 4, None, GL_DYNAMIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, pick_ssbo_id)

def clear_pick_ssbo():
    v = (GLuint * 1)(_PICK_EMPTY)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, pick_ssbo_id)
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 4, v)

def read_pick_ssbo() -> tuple[int, float] | None:
    """`(objectIndex, depth01) | None`"""
    v = (GLuint * 1)(_PICK_EMPTY)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, pick_ssbo_id)
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 4, v)
    packed = v[0]
    if packed == _PICK_EMPTY:
        return None
    idx    = packed & 0xFFFF          # 16 位 id
    depthQ = packed >> 16             # 16 位深度
    return idx, depthQ / float(0xFFFF)

_instance_mat_ssbo: int = -1
_instance_id_ssbo: int = -1
_max_instances: int = 0

def init_instance_buffers(n_max: int):
    global _instance_mat_ssbo, _instance_id_ssbo, _max_instances
    _max_instances = n_max

    ssbos = (GLuint * 2)()
    glGenBuffers(2, ssbos)
    _instance_mat_ssbo = ssbos[0]
    _instance_id_ssbo  = ssbos[1]

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _instance_mat_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, n_max * 64, None, GL_DYNAMIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _instance_mat_ssbo)

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _instance_id_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, n_max * 4, None, GL_DYNAMIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _instance_id_ssbo)

def upload_instances(entries: list[tuple[glm.mat4, int]]):
    """entries: [(model_matrix, object_id), ...]"""
    n = len(entries)
    assert n <= _max_instances, f"capacity {_max_instances}, got {n}"

    mat_data = (GLfloat * (n * 16))()
    id_data  = (GLuint * n)()
    for i, (m, oid) in enumerate(entries):
        # 列主序：内存中第 c*4+r 个 float 是 m[c][r]
        for c in range(4):
            for r in range(4):
                mat_data[i * 16 + c * 4 + r] = m[c][r]
        id_data[i] = oid

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _instance_mat_ssbo)
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * 64, mat_data)

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _instance_id_ssbo)
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * 4, id_data)

SHADOW_NONE  = 0
SHADOW_HARD = 1
SHADOW_PCF  = 2

_shadow_fbo:  int = -1
_shadow_cube: int = -1
_shadow_size: int = -1
_shadow_far:  float = -1.0

def get_shadow_cube():
    return _shadow_cube

def init_shadow_map(size: int = 1024, far_plane: float = 60.0):
    global _shadow_fbo, _shadow_cube, _shadow_size, _shadow_far
    _shadow_size = size
    _shadow_far = far_plane

    texs = (GLuint * 1)()
    glGenTextures(1, texs)
    _shadow_cube = texs[0]

    glBindTexture(GL_TEXTURE_CUBE_MAP, _shadow_cube)
    for i in range(6):
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0,
                     GL_DEPTH_COMPONENT24, size, size, 0,
                     GL_DEPTH_COMPONENT, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

    fbos = (GLuint * 1)()
    glGenFramebuffers(1, fbos)
    _shadow_fbo = fbos[0]

    glBindFramebuffer(GL_FRAMEBUFFER, _shadow_fbo)
    glDrawBuffer(GL_NONE)
    glReadBuffer(GL_NONE)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

def _cube_face_views(light_pos: tuple | glm.vec3) -> list:
    """+X, -X, +Y, -Y, +Z, -Z 六个视图矩阵"""
    p = glm.vec3(*light_pos)
    dirs = [
        (glm.vec3( 1, 0, 0), glm.vec3(0, -1, 0)),
        (glm.vec3(-1, 0, 0), glm.vec3(0, -1, 0)),
        (glm.vec3( 0, 1, 0), glm.vec3(0, 0,  1)),
        (glm.vec3( 0,-1, 0), glm.vec3(0, 0, -1)),
        (glm.vec3( 0, 0, 1), glm.vec3(0, -1, 0)),
        (glm.vec3( 0, 0,-1), glm.vec3(0, -1, 0)),
    ]
    return [glm.lookAt(p, p + d, u) for d, u in dirs]

def render_shadow_cube(
        prog_shadow: ShaderProgram,
        light_pos: tuple | glm.vec3,
        geom_list,
        far_plane=None):
    """geom_list: [(geometry, instance_count, instance_offset, halfs, radius), ...]"""
    if far_plane is None:
        far_plane = _shadow_far

    glBindFramebuffer(GL_FRAMEBUFFER, _shadow_fbo)
    glViewport(0, 0, _shadow_size, _shadow_size)
    glClearColor(0, 0, 0, 0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)

    proj  = glm.perspective(glm.radians(90.0), 1.0, 0.1, far_plane)
    views = _cube_face_views(light_pos)

    prog_shadow.use()
    prog_shadow["lightPos"] = tuple(light_pos)
    prog_shadow["lightFar"] = far_plane
    uniform_matrix(prog_shadow, "lightProj", proj)

    for i in range(6):
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                               GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                               _shadow_cube, 0)
        glClear(GL_DEPTH_BUFFER_BIT)
        uniform_matrix(prog_shadow, "lightView", views[i])

        for geom, count, offset, halfs, rad in geom_list:
            prog_shadow["halfSize"]       = halfs
            prog_shadow["radius"]         = rad
            prog_shadow["instanceOffset"] = offset
            draw_instanced(geom, count)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glViewport(0, 0, win.width, win.height)

_PHYS_STRIDE = 96
_phys_ssbo = -1
_phys_capacity = 0

def init_physics_buffers(n_max: int):
    global _phys_ssbo, _phys_capacity
    _phys_capacity = n_max
    ssbos = (GLuint * 1)()
    glGenBuffers(1, ssbos)
    _phys_ssbo = ssbos[0]
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _phys_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, n_max * _PHYS_STRIDE, None, GL_DYNAMIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _phys_ssbo)

def init_physics_body(idx, position, quat, half_size,
                      mass=1.0,
                      lin_vel=(0,0,0), ang_vel=(0,0,0),
                      restitution=0.15, friction=0.4):
    """均匀盒惯量：I_x = m/3 * (hy^2 + hz^2)，对角。"""
    hx, hy, hz = half_size
    inv_m = 0.0 if mass <= 0 else 1.0 / mass
    Ix = (mass / 3.0) * (hy*hy + hz*hz) if mass > 0 else 1.0
    Iy = (mass / 3.0) * (hx*hx + hz*hz) if mass > 0 else 1.0
    Iz = (mass / 3.0) * (hx*hx + hy*hy) if mass > 0 else 1.0

    d = (GLfloat * 24)()
    d[0:4]   = (position[0], position[1], position[2], inv_m)
    d[4:8]   = (quat[0], quat[1], quat[2], quat[3])
    d[8:12]  = (lin_vel[0], lin_vel[1], lin_vel[2], restitution)
    d[12:16] = (ang_vel[0], ang_vel[1], ang_vel[2], friction)
    d[16:20] = (hx, hy, hz, 0.0)
    d[20:24] = (1.0/Ix, 1.0/Iy, 1.0/Iz, 0.0)

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _phys_ssbo)
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, idx * _PHYS_STRIDE, _PHYS_STRIDE, d)

def step_physics(prog, mat_ssbo, n_bodies, dt,
                 gravity=(0.0, -9.8, 0.0),
                 ground_y=-0.5,
                 ang_damping=0.97,
                 model_scale=0.5,
                 table_half_xz=(10.0, 10.0),
                 water_y=-0.7,
                 pool_floor_y=-3.0,
                 water_lin_drag=0.98,
                 water_ang_drag=0.95,
                 buoyancy_gain=1.0,
                 time=0.0):
    REF_DT = 1.0 / 60.0
    ang_d = ang_damping ** (dt / REF_DT)

    prog.use()
    prog["nBodies"]      = n_bodies
    prog["dt"]           = float(dt)
    prog["gravity"]      = gravity
    prog["groundY"]      = float(ground_y)
    prog["angDamping"]   = ang_d
    prog["modelScale"]   = model_scale
    prog["tableHalfXZ"]  = tuple(table_half_xz)
    prog["time"]         = float(time)

    prog["waterY"]       = float(water_y)
    prog["poolFloorY"]   = float(pool_floor_y)
    prog["waterLinDrag"] = float(water_lin_drag)
    prog["waterAngDrag"] = float(water_ang_drag)
    prog["buoyancyGain"] = float(buoyancy_gain)

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, mat_ssbo)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _phys_ssbo)

    glDispatchCompute((n_bodies + 63) // 64, 1, 1)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

def quat_from_axis_angle(axis, angle_rad):
    a = glm.normalize(glm.vec3(*axis))
    h = angle_rad * 0.5
    s = glm.sin(h)
    return (a.x * s, a.y * s, a.z * s, glm.cos(h))

def set_body_velocity(idx: int,
                      lin_vel=None,
                      ang_vel=None,
                      only_if_slower_than: float = 1e30):
    if lin_vel is None and ang_vel is None:
        return

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _phys_ssbo)

    # 若要判速，先读一下
    if only_if_slower_than < 1e29:
        cur = (GLfloat * 3)()
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER,
                           idx * _PHYS_STRIDE + 32, 12, cur)
        spd = (cur[0]*cur[0] + cur[1]*cur[1] + cur[2]*cur[2]) ** 0.5
        if spd > only_if_slower_than:
            return

    if lin_vel is not None:
        data = (GLfloat * 3)(*lin_vel)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                        idx * _PHYS_STRIDE + 32, 12, data)
    if ang_vel is not None:
        data = (GLfloat * 3)(*ang_vel)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                        idx * _PHYS_STRIDE + 48, 12, data)

def create_water_quad(half_xz: float, y: float):
    """在 y 高度生成一个 [-half_xz, +half_xz]² 的水平四边形。"""
    verts = [
        -half_xz, y, -half_xz,
         half_xz, y, -half_xz,
         half_xz, y,  half_xz,
        -half_xz, y, -half_xz,
         half_xz, y,  half_xz,
        -half_xz, y,  half_xz,
    ]
    vbos = (GLuint * 1)()
    glGenBuffers(1, vbos)
    vbo = vbos[0]
    data = (GLfloat * len(verts))(*verts)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)
    return vbo, 6

def draw_water(vbo: int, n_verts: int):
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), None)
    glEnableVertexAttribArray(0)
    glDrawArrays(GL_TRIANGLES, 0, n_verts)

def get_body_state(idx: int):
    """
    返回 `(pos: glm.vec3, quat: tuple, lin_vel: glm.vec3, ang_vel: glm.vec3)`

    会 `glBindBuffer(GL_SHADER_STORAGE_BUFFER)` 用完不还原
    """
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _phys_ssbo)
    d = (GLfloat * 16)()
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER,
                       idx * _PHYS_STRIDE, 64, d)

    pos = glm.vec3(d[0], d[1], d[2])
    quat = (d[4], d[5], d[6], d[7])
    lin = glm.vec3(d[8],  d[9],  d[10])
    ang = glm.vec3(d[12], d[13], d[14])
    return pos, quat, lin, ang
