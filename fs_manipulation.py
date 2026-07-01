import numpy as np
import struct, os, tempfile, uuid, ctypes

def read_fs(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        # Заголовок
        magic = f.read(8)
        if magic != b'KSFV_01\x00':
            raise ValueError(f"Неверный magic: {magic!r}")
        _, n, h, w = struct.unpack('<IIII', f.read(16))

        if n == 0 or h == 0 or w == 0:
            raise ValueError(f"Некорректные размеры: N={n} H={h} W={w}")

        # Кадры
        n_bytes = n * h * w * 4
        raw = f.read(n_bytes)
        if len(raw) < n_bytes:
            raise ValueError(f"Файл обрезан: ожидалось {n_bytes} байт кадров")

        frames = np.frombuffer(raw, dtype='<f4').reshape(n, h, w).transpose(1, 2, 0).copy()
        # Байты после кадров — OLE CFB (не читается Python-кодом напрямую)

    return frames

### Functions to save fs

EMPTY_VECT = struct.pack('<I', 0)

def ser_seqinfo() -> bytes:
    buf = bytearray(288)
    buf[0] = 1; buf[1] = 1       # bSubtrBG=true, bUseHeatDur=true
    struct.pack_into('<10d', buf, 8,
        0.0, 0.0, 0.0, 0.0,      # bgInd, heatStartInd, calcStartInd, calcStopInd
        0.001, 0.1,              # heatingTime, timeStep
        6e-3, 5.2e-7, 0.64,      # sampThick, sampDiff, sampCond
        0.001,                   # mPixRatio
    )
    return bytes(buf)

def ser_kivect(frames: np.ndarray) -> bytes:
    N = frames.shape[0]
    buf = bytearray(4 + N * 16)
    struct.pack_into('<I', buf, 0, N)
    for i, fr in enumerate(frames):
        struct.pack_into('<dd', buf, 4 + i * 16,
                         float(np.min(fr)), float(np.max(fr)))
    return bytes(buf)

def ser_palette_oc() -> bytes:
    # SerialStrings["Gray"]: sizes-store + string
    ser_str = struct.pack('<II', 1, 4) + b'Gray'   # 12 байт

    # m_ColorsCnt = 250
    col_cnt = struct.pack('<I', 250)               # 4 байта

    # MakeHalfTone(250): color[i]=round(i*255/250) для i<250, остальные clBlue
    colors = []
    for i in range(250):
        c = round(i * 255.0 / 250)
        colors.append(c | (c << 8) | (c << 16))   # RGB(c,c,c)
    for _ in range(6):
        colors.append(0x00FF0000)                  # clBlue в VCL

    sto = struct.pack('<I', 256) + struct.pack('<256I', *colors)  # 1028 байт
    return ser_str + col_cnt + sto

def ser_axisinfo() -> bytes:
    return struct.pack('<IIIII', 2, 0, 0, 0, 0)  # 20 байт

def ser_kvinfo(frames: np.ndarray) -> bytes:
    buf = bytearray(88)
    struct.pack_into('<d', buf, 8, 1.0)    # m_ScaleX
    struct.pack_into('<d', buf, 16, 1.0)   # m_ScaleY
    g_min = float(frames.min())
    g_max = float(frames.max())
    struct.pack_into('<dd', buf, 24, g_min, g_max)   # m_TotalViewRange
    struct.pack_into('<dd', buf, 40, g_min, g_max)   # m_ViewRange
    buf[72] = 1; buf[73] = 1              # bAutoMarg, bRangeByFrame
    clWhite = 0x00FFFFFF
    struct.pack_into('<III', buf, 76, clWhite, clWhite, clWhite)
    return bytes(buf)

def save_fs(filepath: str, frames: np.ndarray):
    frames = frames.transpose(2, 0, 1) # (h, w, t) -> (t, h, w)

    N, H, W = frames.shape

    streams = {
        'SeqInfo':      ser_seqinfo(),
        'KIVect':       ser_kivect(frames),
        'KVInfo':       ser_kvinfo(frames),
        'Pal':          ser_palette_oc(),
        'IsoTerm':      EMPTY_VECT,
        'AddPal':       EMPTY_VECT,
        'TxtInfo':      EMPTY_VECT,
        'DescInfo':     EMPTY_VECT,
        'TDTool':       EMPTY_VECT,
        'TDLabel':      EMPTY_VECT,
        'TDCV':         EMPTY_VECT,
        'SLabel':       EMPTY_VECT,
        'Arrow':        EMPTY_VECT,
        'AxisInfo':     ser_axisinfo(),
        'XAxisData':    EMPTY_VECT,
        'HeatOnTimes':  EMPTY_VECT,
        'HeatOffTimes': EMPTY_VECT,
    }

    # Временный OLE CFB файл
    ksi_path = os.path.join(tempfile.gettempdir(),
                            f'ksi_{uuid.uuid4().hex}.ksi')
    try:
        write_ksi_ole(ksi_path, streams)
        with open(ksi_path, 'rb') as f:
            ksi_bytes = f.read()
    finally:
        try: os.unlink(ksi_path)
        except OSError: pass

    with open(filepath, 'wb') as f:
        # Заголовок (24 байта)
        f.write(b'KSFV_01\x00')
        f.write(struct.pack('<III', 0, N, H))
        f.write(struct.pack('<I', W))

        # Кадры (N × H × W × 4 байта)
        for frame in frames:
            f.write(np.asarray(frame, dtype='<f4').tobytes())

        # OLE CFB метаданных
        f.write(ksi_bytes)
    
def write_ksi_ole(ksi_path: str, streams: dict) -> None:
    """Записывает OLE CFB с потоками по пути ksi_path.

    streams: {имя_потока: bytes_данные}
    """
    ole32 = ctypes.windll.ole32
    STGM_CREATE = 0x1000; STGM_WRITE = 0x1; STGM_READWRITE = 0x2
    STGM_SHARE_EXCLUSIVE = 0x10

    ole32.StgCreateDocfile.restype  = ctypes.c_long
    ole32.StgCreateDocfile.argtypes = [
        ctypes.c_wchar_p, ctypes.c_ulong, ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    stg = ctypes.c_void_p()
    hr = ole32.StgCreateDocfile(ksi_path,
                                STGM_CREATE | STGM_READWRITE | STGM_SHARE_EXCLUSIVE,
                                0, ctypes.byref(stg))
    if hr != 0:
        raise OSError(f"StgCreateDocfile: 0x{hr & 0xFFFFFFFF:08X}")

    stg_vtbl = ctypes.cast(
        ctypes.cast(stg, ctypes.POINTER(ctypes.c_void_p))[0],
        ctypes.POINTER(ctypes.c_void_p))

    # IStorage vtable: [2]=Release, [3]=CreateStream, [9]=Commit
    CreateStream_t = ctypes.WINFUNCTYPE(
        ctypes.c_long, ctypes.c_void_p, ctypes.c_wchar_p,
        ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_void_p))
    Commit_t  = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_ulong)
    Release_t = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)

    create_stream = CreateStream_t(stg_vtbl[3])
    commit_stg    = Commit_t(stg_vtbl[9])
    release_stg   = Release_t(stg_vtbl[2])

    # IStream vtable: [2]=Release, [4]=Write
    Write_t    = ctypes.WINFUNCTYPE(
        ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong))
    Release_t2 = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)

    stream_mode = STGM_CREATE | STGM_WRITE | STGM_SHARE_EXCLUSIVE
    try:
        for name, data in streams.items():
            stm = ctypes.c_void_p()
            hr = create_stream(stg.value, name, stream_mode, 0, 0, ctypes.byref(stm))
            if hr != 0:
                raise OSError(f"CreateStream({name!r}): 0x{hr & 0xFFFFFFFF:08X}")
            stm_vtbl = ctypes.cast(
                ctypes.cast(stm, ctypes.POINTER(ctypes.c_void_p))[0],
                ctypes.POINTER(ctypes.c_void_p))
            write_stm   = Write_t(stm_vtbl[4])
            release_stm = Release_t2(stm_vtbl[2])
            try:
                if data:
                    buf     = (ctypes.c_uint8 * len(data))(*data)
                    written = ctypes.c_ulong(0)
                    hr = write_stm(stm.value, buf, len(data), ctypes.byref(written))
                    if hr != 0:
                        raise OSError(f"IStream::Write({name!r}): 0x{hr & 0xFFFFFFFF:08X}")
            finally:
                release_stm(stm.value)
        commit_stg(stg.value, 0)
    finally:
        release_stg(stg.value)