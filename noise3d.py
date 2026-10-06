from random import Random
from pyglet.gl import *
from pyglm import glm


class Noise3D:
    """3D 噪声纹理基类。

    子类通过重写 `compute_pix` 决定每个体素的 RGBA 值。
    """

    def __init__(self, size: tuple | glm.ivec3, seed: int | None = None):
        texs = (GLuint * 1)()
        glGenTextures(1, texs)
        self.id: int = texs[0]

        rng = Random(seed)
        self._noise = [
            rng.random() for _ in range(size[0] * size[1] * size[2])
        ]
        self.data = []

        for i in range(size[0]):
            for j in range(size[1]):
                for k in range(size[2]):
                    rgba = self.compute_pix(i, j, k)
                    self.data.extend(rgba)

        glBindTexture(GL_TEXTURE_3D, self.id)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexStorage3D(GL_TEXTURE_3D, 1, GL_RGBA8, *size)
        arr = (GLubyte * len(self.data))(*self.data)
        glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, *size, GL_RGBA,
                        GL_UNSIGNED_INT_8_8_8_8_REV, arr)

    def compute_pix(self, i, j, k) -> glm.ivec4:
        raise NotImplementedError

class WaterNoise3D(Noise3D):
    def __init__(
        self,
        size: tuple | glm.ivec3,
        seed: int | None = 0,
        wave_freq_x: float = 1.0,
        wave_freq_z: float = 1.0,
        wave_amp: float = 0.6,
        noise_amp: float = 0.4,
    ):
        # 下面这些字段必须早于 super().__init__ 设置 —— 基类会立刻调用 compute_pix
        self._size = (int(size[0]), int(size[1]), int(size[2]))
        self._wave_freq_x = float(wave_freq_x)
        self._wave_freq_z = float(wave_freq_z)
        self._wave_amp = float(wave_amp)
        self._noise_amp = float(noise_amp)

        super().__init__(self._size, seed=seed)

    def compute_pix(self, i: int, j: int, k: int):
        sx, sy, sz = self._size

        n = self._noise[(i * sy + j) * sz + k]

        u = i / max(sx - 1, 1)
        v = k / max(sz - 1, 1)

        wave = glm.sin(
            6.2831 * (self._wave_freq_x * u + self._wave_freq_z * v)
        )
        wave = 0.5 + 0.5 * wave # [-1, 1] -> [0, 1]

        val = self._wave_amp * wave + self._noise_amp * n
        val = 0.0 if val < 0.0 else (1.0 if val > 1.0 else val)
        g = int(val * 255.0 + 0.5)

        return g, g, g, 255
