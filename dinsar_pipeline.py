import os
import glob
import subprocess
import datetime
import zipfile
import re
import shutil
import logging
import argparse
import json
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from math import cos, radians

import geopandas as gpd
import psutil
from shapely.geometry import Point

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.merge import merge

from concurrent.futures import ProcessPoolExecutor, as_completed
from logging.handlers import RotatingFileHandler
from typing import Optional, List, Tuple, Dict, Any

@dataclass(frozen=True)
class PipelineConfig:
    input_dir: str
    shp_path: str
    project_root: str
    output_base_dir: str
    output_dir: str
    temp_dir: str
    gpt_path: str
    snaphu_cmd: str
    max_cpu_tasks: int
    threads_per_worker: int
    jvm_heap: str
    tile_cache: str
    incidence_angle: float
    coherence_threshold: float
    weighted_stacking: bool
    orbital_ramp_removal: bool
    topo_phase_correction: bool
    dem_tif_path: str
    dem_name: str
    use_esd: bool
    output_mm: bool
    required_disk_gb: int
    max_temporal_baseline_days: int
    loop_closure_threshold_mm: float
    atm_filter_block_size: int
    jump_gradient_threshold: float
    robust_estimation: bool
    robust_mad_threshold: float
    reference_coherence_percentile: float


# ================= 服务器配置（按你给的） =================
DEFAULT_INPUT_DIR = r"D:\leixiang\D-InSAR\S1_Data"
DEFAULT_SHP_PATH = r"D:\leixiang\D-InSAR\GreatWall_Buffer\Hebei_Baoding_1km.shp"
DEFAULT_PROJECT_ROOT = r"D:\leixiang\D-InSAR"

DEFAULT_GPT_PATH = r"C:\Program Files\esa-snap\bin\gpt.exe"
DEFAULT_SNAPHU_CMD = r"D:\leixiang\D-InSAR\Software\snaphu-v1.4.2_win64\bin\snaphu.exe"

DEFAULT_MAX_CPU_TASKS = 2  # 并行 gpt/snaphu 的进程数（先跑通可改 1）
DEFAULT_THREADS_PER_WORKER = 40  # 每个 GPT 进程线程
DEFAULT_JVM_HEAP = "32G"  # 每个 GPT 进程堆
DEFAULT_TILE_CACHE = "12G"
DEFAULT_REQUIRED_DISK_GB = 500
DEFAULT_MAX_TEMPORAL_BASELINE_DAYS = 60
DEFAULT_LOOP_CLOSURE_THRESHOLD_MM = 15.0
DEFAULT_ATM_FILTER_BLOCK_SIZE = 32
DEFAULT_JUMP_GRADIENT_THRESHOLD = 0.08
DEFAULT_ROBUST_ESTIMATION = True
DEFAULT_ROBUST_MAD_THRESHOLD = 3.0
DEFAULT_REFERENCE_COHERENCE_PERCENTILE = 95.0

DEFAULT_INCIDENCE_ANGLE = 39.5
DEFAULT_COHERENCE_THRESHOLD = 0.30
DEFAULT_WEIGHTED_STACKING = True  # 相干性加权堆栈
DEFAULT_ORBITAL_RAMP_REMOVAL = True  # 轨道误差精炼（移除平面相位坡度）
DEFAULT_TOPO_PHASE_CORRECTION = True  # 大气延迟线性修正（与高程线性相关）
DEFAULT_DEM_TIF_PATH = r"D:\leixiang\D-InSAR\DEM\dem.tif"

# 精度向选项
DEFAULT_DEM_NAME = "Copernicus 30m Global DEM"  # 想和本地一致就改回 "SRTM 3Sec"
DEFAULT_USE_ESD = True  # TOPS 配准增强
DEFAULT_OUTPUT_MM = True  # 输出毫米（PhaseToDisplacement 通常是米）
# =========================================================


LOGGER = logging.getLogger(__name__)


def setup_logging(log_dir: str, run_id: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    log_path = os.path.join(log_dir, f"pipeline_{run_id}.log")
    file_handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


XML_PREPROCESS = r"""<graph id="Graph">
  <version>1.0</version>
  <node id="Read"><operator>Read</operator>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>${input}</file>
    </parameters>
  </node>

  <node id="Split"><operator>TOPSAR-Split</operator>
    <sources><sourceProduct refid="Read"/></sources>
    <parameters>
      <subswath>${subswath}</subswath>
      <selectedPolarisations>VV</selectedPolarisations>
      <firstBurstIndex>${startBurst}</firstBurstIndex>
      <lastBurstIndex>${endBurst}</lastBurstIndex>
    </parameters>
  </node>

  <node id="Apply-Orbit"><operator>Apply-Orbit-File</operator>
    <sources><sourceProduct refid="Split"/></sources>
    <parameters>
      <orbitType>Sentinel Precise (Auto Download)</orbitType>
      <polyDegree>3</polyDegree>
      <continueOnFail>false</continueOnFail>
    </parameters>
  </node>

  <node id="Write"><operator>Write</operator>
    <sources><sourceProduct refid="Apply-Orbit"/></sources>
    <parameters><file>${output}</file><formatName>BEAM-DIMAP</formatName></parameters>
  </node>
</graph>"""

# Step1：核心干涉（ReadM + ReadS + BackGeocoding 双输入）
XML_STEP1_CORE = r"""<graph id="Graph">
  <version>1.0</version>

  <node id="ReadM"><operator>Read</operator>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement"><file>${master}</file></parameters>
  </node>
  <node id="ReadS"><operator>Read</operator>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement"><file>${slave}</file></parameters>
  </node>

  <node id="Back-Geocoding"><operator>Back-Geocoding</operator>
    <sources>
      <sourceProduct refid="ReadM"/>
      <sourceProduct.1 refid="ReadS"/>
    </sources>
    <parameters>
      <demName>${demName}</demName>
      <demResamplingMethod>BICUBIC_INTERPOLATION</demResamplingMethod>
      <resamplingType>BICUBIC_INTERPOLATION</resamplingType>
      <maskOutAreaWithoutElevation>true</maskOutAreaWithoutElevation>
    </parameters>
  </node>

  ${ESD_NODE}

  <node id="Interferogram"><operator>Interferogram</operator>
    <sources><sourceProduct refid="${PRE_IFG_NODE}"/></sources>
    <parameters>
      <subtractFlatEarthPhase>true</subtractFlatEarthPhase>
      <srpPolynomialDegree>5</srpPolynomialDegree>
      <srpNumberPoints>501</srpNumberPoints>
      <orbitDegree>3</orbitDegree>
      <includeCoherence>true</includeCoherence>
    </parameters>
  </node>

  <node id="Deburst"><operator>TOPSAR-Deburst</operator>
    <sources><sourceProduct refid="Interferogram"/></sources>
  </node>

  <node id="Write"><operator>Write</operator>
    <sources><sourceProduct refid="Deburst"/></sources>
    <parameters><file>${output}</file><formatName>BEAM-DIMAP</formatName></parameters>
  </node>
</graph>"""

# Step2：Multilook + TopoRemoval + Goldstein + Subset + SnaphuExport
XML_STEP2_FILTER = r"""<graph id="Graph">
  <version>1.0</version>

  <node id="Read"><operator>Read</operator>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>${input}</file><formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>

  <node id="Multilook"><operator>Multilook</operator>
    <sources><sourceProduct refid="Read"/></sources>
    <parameters>
      <nRgLooks>8</nRgLooks>
      <nAzLooks>2</nAzLooks>
      <outputIntensity>false</outputIntensity>
      <grSquarePixel>true</grSquarePixel>
    </parameters>
  </node>

  <node id="TopoPhaseRemoval"><operator>TopoPhaseRemoval</operator>
    <sources><sourceProduct refid="Multilook"/></sources>
    <parameters><demName>${demName}</demName></parameters>
  </node>

  <node id="GoldsteinFilter"><operator>GoldsteinPhaseFiltering</operator>
    <sources><sourceProduct refid="TopoPhaseRemoval"/></sources>
    <parameters>
      <alpha>0.8</alpha>
      <FFTSizeString>64</FFTSizeString>
      <windowSizeString>3</windowSizeString>
      <useCoherenceMask>false</useCoherenceMask>
    </parameters>
  </node>

  <node id="Subset"><operator>Subset</operator>
    <sources><sourceProduct refid="GoldsteinFilter"/></sources>
    <parameters><geoRegion>${wkt}</geoRegion><copyMetadata>true</copyMetadata></parameters>
  </node>

  <node id="SnaphuExport"><operator>SnaphuExport</operator>
    <sources><sourceProduct refid="Subset"/></sources>
    <parameters>
      <targetFolder>${targetFolder}</targetFolder>
      <statCostMode>SMOOTH</statCostMode>
      <initMethod>MST</initMethod>
      <numberOfTileCols>10</numberOfTileCols>
      <numberOfTileRows>10</numberOfTileRows>
      <tileCostThreshold>500</tileCostThreshold>
    </parameters>
  </node>

  <node id="Write"><operator>Write</operator>
    <sources><sourceProduct refid="Subset"/></sources>
    <parameters><file>${output}</file><formatName>BEAM-DIMAP</formatName></parameters>
  </node>
</graph>"""

# Step3：SnaphuImport + PhaseToDisplacement + TerrainCorrection
XML_GEO_DISP = r"""<graph id="Graph"><version>1.0</version>
  <node id="ReadW"><operator>Read</operator>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement"><file>${wrapped}</file></parameters>
  </node>
  <node id="ReadU"><operator>Read</operator>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement"><file>${unwrapped}</file><formatName>Snaphu</formatName></parameters>
  </node>

  <node id="SnaphuImport"><operator>SnaphuImport</operator>
    <sources>
      <sourceProduct refid="ReadW"/>
      <sourceProduct.1 refid="ReadU"/>
    </sources>
    <parameters><doNotKeepWrapped>false</doNotKeepWrapped></parameters>
  </node>

  <node id="PhaseToDisplacement"><operator>PhaseToDisplacement</operator>
    <sources><sourceProduct refid="SnaphuImport"/></sources>
  </node>

  <node id="Terrain-Correction"><operator>Terrain-Correction</operator>
    <sources><sourceProduct refid="PhaseToDisplacement"/></sources>
    <parameters>
      <demName>${demName}</demName>
      <imgResamplingMethod>BILINEAR_INTERPOLATION</imgResamplingMethod>
      <saveSelectedSourceBand>true</saveSelectedSourceBand>
    </parameters>
  </node>

  <node id="Write"><operator>Write</operator>
    <sources><sourceProduct refid="Terrain-Correction"/></sources>
    <parameters><file>${outFile}</file><formatName>GeoTIFF</formatName></parameters>
  </node>
</graph>"""

XML_GEO_COH = r"""<graph id="Graph"><version>1.0</version>
  <node id="Read"><operator>Read</operator>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement"><file>${input}</file><formatName>BEAM-DIMAP</formatName></parameters>
  </node>

  <node id="Terrain-Correction"><operator>Terrain-Correction</operator>
    <sources><sourceProduct refid="Read"/></sources>
    <parameters>
      <demName>${demName}</demName>
      <imgResamplingMethod>BILINEAR_INTERPOLATION</imgResamplingMethod>
      <saveSelectedSourceBand>true</saveSelectedSourceBand>
      <sourceBands>${cohBand}</sourceBands>
    </parameters>
  </node>

  <node id="Write"><operator>Write</operator>
    <sources><sourceProduct refid="Terrain-Correction"/></sources>
    <parameters><file>${outFile}</file><formatName>GeoTIFF</formatName></parameters>
  </node>
</graph>"""


def validate_environment(config: PipelineConfig) -> None:
    missing = []
    for path in [config.input_dir, config.shp_path, config.gpt_path, config.snaphu_cmd]:
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        missing_str = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(f"缺少必要路径/可执行文件:\n{missing_str}")
    if config.topo_phase_correction and not os.path.exists(config.dem_tif_path):
        LOGGER.warning("未找到 DEM_TIF_PATH: %s，将跳过线性高程修正", config.dem_tif_path)
    check_disk_space(config.project_root, config.required_disk_gb)


def check_disk_space(path: str, required_gb: int) -> None:
    if not os.path.exists(path):
        path = os.path.dirname(path)
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    if free_gb < required_gb:
        raise RuntimeError(
            f"磁盘剩余空间不足: {free_gb:.1f} GB < {required_gb} GB"
        )


def create_xml_file(template: str, replace_dict: Dict[str, str], out_path: str) -> None:
    content = template
    for k, v in replace_dict.items():
        content = content.replace("${" + k + "}", str(v).replace("\\", "/"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)


def run_gpt(
    config: PipelineConfig,
    xml_path: str,
    task_id: str,
    work_dir: Optional[str] = None,
    max_retries: int = 1,
    retry_sleep_s: int = 10,
) -> None:
    os.makedirs(config.temp_dir, exist_ok=True)
    base_tmp = os.path.abspath(os.path.join(config.project_root, "vcp_tmp"))
    local_tmp = os.path.abspath(os.path.join(base_tmp, task_id))
    os.makedirs(local_tmp, exist_ok=True)
    time.sleep(0.2)

    java_tmp = local_tmp.replace("\\", "/").rstrip("/")
    env = os.environ.copy()
    env["JAVA_TOOL_OPTIONS"] = (
        f"-Xmx{config.jvm_heap} -Djava.io.tmpdir={java_tmp} -XX:+UseG1GC"
    )
    env["TMPDIR"] = java_tmp
    env["TMP"] = java_tmp
    env["TEMP"] = java_tmp
    normalize_env_paths(env)

    log_path = os.path.join(config.temp_dir, f"gpt_{task_id}.log")
    cmd = [
        config.gpt_path,
        xml_path,
        "-q",
        str(config.threads_per_worker),
        "-c",
        config.tile_cache,
        "-x",
    ]

    attempt = 0
    while True:
        attempt += 1
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n=== GPT attempt {attempt}/{max_retries + 1} ===\n")
            p = subprocess.run(
                cmd,
                cwd=work_dir,
                env=env,
                stdout=log,
                stderr=log,
                check=False,
            )

        if p.returncode == 0:
            break
        if attempt > max_retries:
            break
        LOGGER.warning(
            "GPT 失败(%s) attempt %s/%s，%ss 后重试",
            task_id,
            attempt,
            max_retries + 1,
            retry_sleep_s,
        )
        time.sleep(retry_sleep_s)

    shutil.rmtree(local_tmp, ignore_errors=True)

    if p.returncode != 0:
        raise RuntimeError(
            f"GPT 失败({task_id})，退出码 {p.returncode}，查看日志: {log_path}"
        )


def load_status(status_path: str) -> Dict[str, Any]:
    if not os.path.exists(status_path):
        return {}
    try:
        with open(status_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_status(status_path: str, data: Dict[str, Any]) -> None:
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def compute_stats(arr: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(arr)
    if not np.any(valid):
        return {"valid_ratio": 0.0}
    data = arr[valid]
    return {
        "valid_ratio": float(valid.mean()),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
    }


def parse_size_to_bytes(size_value: str) -> int:
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([KMGTP]?B?)\s*$", size_value, re.I)
    if not match:
        raise ValueError(f"无法解析大小字符串: {size_value}")
    value = float(match.group(1))
    unit = match.group(2).upper()
    if unit in ("K", "KB"):
        factor = 1024
    elif unit in ("M", "MB"):
        factor = 1024**2
    elif unit in ("G", "GB", ""):
        factor = 1024**3
    elif unit in ("T", "TB"):
        factor = 1024**4
    else:
        raise ValueError(f"未知单位: {unit}")
    return int(value * factor)


def normalize_tile_cache(jvm_heap: str, tile_cache: str) -> str:
    heap_bytes = parse_size_to_bytes(jvm_heap)
    cache_bytes = parse_size_to_bytes(tile_cache)
    if cache_bytes >= heap_bytes:
        adjusted = int(heap_bytes * 0.6)
        adjusted_gb = max(adjusted // (1024**3), 1)
        return f"{adjusted_gb}G"
    return tile_cache


def bytes_to_gb(value: float) -> int:
    return max(int(value // (1024**3)), 1)


def compute_memory_settings(max_workers: int) -> Tuple[str, str, int]:
    available_bytes = psutil.virtual_memory().available
    per_task_bytes = max(available_bytes * 0.4, 4 * 1024**3)
    dynamic_max = max(1, int(available_bytes // per_task_bytes))
    adjusted_max_workers = min(max_workers, dynamic_max)
    jvm_heap_gb = max(int(per_task_bytes // (1024**3)), 4)
    tile_cache_gb = max(int(jvm_heap_gb * 0.6), 1)
    return f"{jvm_heap_gb}G", f"{tile_cache_gb}G", adjusted_max_workers


def normalize_env_paths(env: Dict[str, str]) -> None:
    for key in ("PATH", "JBLAS_HOME", "JBLAS_PATH"):
        if key in env and env[key]:
            env[key] = env[key].replace("\\", "/")


def read_incidence_angle(tif_path: str, fallback: float) -> float:
    try:
        with rasterio.open(tif_path) as src:
            tags = src.tags()
        for key in ("incidence_angle", "incidenceAngle", "INCIDENCE_ANGLE"):
            if key in tags:
                return float(tags[key])
    except (ValueError, rasterio.errors.RasterioIOError):
        return fallback
    return fallback


def read_perpendicular_baseline(tif_path: str) -> float:
    try:
        with rasterio.open(tif_path) as src:
            tags = src.tags()
        for key in ("perpendicular_baseline", "B_PERP", "b_perp", "BPERP"):
            if key in tags:
                return float(tags[key])
    except (ValueError, rasterio.errors.RasterioIOError):
        return 0.0
    return 0.0


def select_reference_point(
    coherence_stack: List[np.ndarray],
    dem: Optional[np.ndarray],
    coherence_percentile: float,
) -> Optional[Tuple[int, int]]:
    if not coherence_stack:
        return None
    mean_coh = np.nanmean(np.stack(coherence_stack, axis=0), axis=0)
    if dem is None:
        threshold = np.nanpercentile(mean_coh, coherence_percentile)
        candidates = np.where(mean_coh >= threshold)
        if candidates[0].size == 0:
            return None
        return int(candidates[0][0]), int(candidates[1][0])

    grad_y, grad_x = np.gradient(dem)
    slope = np.hypot(grad_x, grad_y)
    threshold = np.nanpercentile(mean_coh, coherence_percentile)
    candidates = np.where(mean_coh >= threshold)
    if candidates[0].size == 0:
        return None
    cand_slopes = slope[candidates]
    idx = int(np.nanargmin(cand_slopes))
    return int(candidates[0][idx]), int(candidates[1][idx])


def apply_atmospheric_filter(
    disp: np.ndarray, mask: np.ndarray, block_size: int
) -> np.ndarray:
    if block_size <= 1:
        return disp
    h, w = disp.shape
    bh = max(1, block_size)
    bw = max(1, block_size)
    pad_h = (bh - (h % bh)) % bh
    pad_w = (bw - (w % bw)) % bw
    padded = np.pad(disp, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=np.nan)
    padded_mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)
    reshaped = padded.reshape((padded.shape[0] // bh, bh, padded.shape[1] // bw, bw))
    mask_reshaped = padded_mask.reshape(
        (padded_mask.shape[0] // bh, bh, padded_mask.shape[1] // bw, bw)
    )
    block_vals = np.where(mask_reshaped, reshaped, np.nan)
    block_median = np.nanmedian(block_vals, axis=(1, 3))
    expanded = np.repeat(np.repeat(block_median, bh, axis=0), bw, axis=1)
    expanded = expanded[:h, :w]
    return disp - expanded


def build_reliability_mask(
    disp: np.ndarray,
    coherence: Optional[np.ndarray],
    coherence_threshold: float,
    gradient_threshold: float,
) -> np.ndarray:
    mask = np.ones(disp.shape, dtype=bool)
    if coherence is not None:
        mask &= coherence >= coherence_threshold
    grad_x = np.abs(disp - np.roll(disp, 1, axis=1))
    grad_y = np.abs(disp - np.roll(disp, 1, axis=0))
    grad = np.maximum(grad_x, grad_y)
    mask &= grad <= gradient_threshold
    return mask


def robust_weighted_stack(
    disp_stack: np.ndarray,
    weight_stack: np.ndarray,
    mad_threshold: float,
) -> np.ndarray:
    median = np.nanmedian(disp_stack, axis=0)
    mad = np.nanmedian(np.abs(disp_stack - median), axis=0)
    mad = np.where(mad == 0, np.nan, mad)
    deviation = np.abs(disp_stack - median)
    valid = deviation <= (mad_threshold * mad)
    weights = np.where(valid, weight_stack, 0.0)
    weighted_sum = np.nansum(disp_stack * weights, axis=0)
    weight_sum = np.nansum(weights, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(weight_sum > 0, weighted_sum / weight_sum, np.nan)


def remove_planar_ramp(
    disp: np.ndarray, mask: np.ndarray, enabled: bool
) -> np.ndarray:
    if not enabled:
        return disp
    rows, cols = np.indices(disp.shape)
    valid = mask & np.isfinite(disp)
    if valid.sum() < 10:
        LOGGER.warning("有效像素不足，跳过轨道坡度移除")
        return disp
    x = cols[valid].ravel()
    y = rows[valid].ravel()
    z = disp[valid].ravel()
    A = np.column_stack([x, y, np.ones_like(x)])
    coeff, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    ramp = (coeff[0] * cols + coeff[1] * rows + coeff[2]).astype(disp.dtype)
    return disp - ramp


def linear_topo_phase_correction(
    disp: np.ndarray, dem: np.ndarray, mask: np.ndarray, enabled: bool
) -> np.ndarray:
    if not enabled:
        return disp
    valid = mask & np.isfinite(disp) & np.isfinite(dem)
    if valid.sum() < 10:
        LOGGER.warning("有效像素不足，跳过高程相关线性修正")
        return disp
    x = dem[valid].ravel()
    y = disp[valid].ravel()
    slope, intercept = np.polyfit(x, y, 1)
    correction = (slope * dem + intercept).astype(disp.dtype)
    return disp - correction


def get_date_from_zip(zip_path: str) -> Optional[datetime.datetime]:
    m = re.search(r"(\d{8})T", os.path.basename(zip_path))
    if not m:
        return None
    return datetime.datetime.strptime(m.group(1), "%Y%m%d")


def analyze_all_subswaths(zip_path: str, shp_path: str):
    gdf = gpd.read_file(shp_path)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)
    try:
        aoi_poly = gdf.union_all()
    except Exception:
        aoi_poly = gdf.unary_union

    wkt = aoi_poly.wkt
    if not wkt:
        raise ValueError("无法从 SHP 生成 WKT")

    aoi_buffered = aoi_poly.buffer(0.05)
    aoi_bounds = aoi_buffered.bounds

    tasks = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for swath in ["iw1", "iw2", "iw3"]:
            xml_pattern = re.compile(rf"s1.-{swath}-slc-vv-.*\.xml", re.IGNORECASE)
            target_xml_str = None
            for filename in z.namelist():
                if "annotation/" in filename and xml_pattern.search(filename.split("/")[-1]):
                    with z.open(filename) as f:
                        target_xml_str = f.read().decode("utf-8")
                    break
            if not target_xml_str:
                continue

            root = ET.fromstring(target_xml_str)
            lpb = None
            for elem in root.iter():
                if "linesPerBurst" in elem.tag:
                    lpb = int(elem.text)
                    break
            if lpb is None:
                continue

            grid_points = []
            for elem in root.iter():
                if "geolocationGridPoint" in elem.tag:
                    lat, lon, line = None, None, None
                    for child in elem:
                        if "latitude" in child.tag:
                            lat = float(child.text)
                        if "longitude" in child.tag:
                            lon = float(child.text)
                        if "line" in child.tag:
                            line = int(child.text)
                    if (
                        lat is not None
                        and (aoi_bounds[1] < lat < aoi_bounds[3])
                        and (aoi_bounds[0] < lon < aoi_bounds[2])
                    ):
                        grid_points.append({"line": line, "geometry": Point(lon, lat)})

            if not grid_points:
                continue

            points_gdf = gpd.GeoDataFrame(grid_points, crs="EPSG:4326")
            points_inside = points_gdf[points_gdf.within(aoi_buffered)]
            if points_inside.empty:
                continue

            lines = points_inside["line"].tolist()
            s = max(1, int(min(lines) // lpb))
            e = int(max(lines) // lpb) + 3
            tasks.append({"swath": swath.upper(), "s": s, "e": e})

    tasks.sort(key=lambda item: item["swath"])
    return tasks, wkt


def validate_input_zip(zip_path: str) -> None:
    try:
        with zipfile.ZipFile(zip_path) as zip_file:
            bad_member = zip_file.testzip()
            if bad_member:
                raise zipfile.BadZipFile(f"坏文件成员: {bad_member}")
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"ZIP 文件损坏: {zip_path} ({exc})") from exc


def build_sbas_pairs(
    dated_files: List[Tuple[datetime.datetime, str]], max_baseline_days: int
) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    pairs = []
    for i, (m_date, _) in enumerate(dated_files):
        for s_date, _ in dated_files[i + 1 :]:
            if (s_date - m_date).days <= max_baseline_days:
                pairs.append((m_date, s_date))
            else:
                break
    return pairs


def load_disp_for_reference(
    disp_path: str,
    ref_shape: Tuple[int, int],
    ref_transform,
    ref_crs,
    incidence_angle: float,
) -> np.ndarray:
    with rasterio.open(disp_path) as src:
        disp = np.full(ref_shape, np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=disp,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear,
        )
    angle_cos = cos(radians(incidence_angle))
    if angle_cos == 0:
        angle_cos = cos(radians(DEFAULT_INCIDENCE_ANGLE))
    return disp / angle_cos


def preload_disp_cache(
    pairs: List[Tuple[datetime.datetime, datetime.datetime, str, Optional[str]]]
) -> Dict[Tuple[datetime.datetime, datetime.datetime], np.ndarray]:
    if not pairs:
        return {}
    with rasterio.open(pairs[0][2]) as src:
        ref_shape = (src.height, src.width)
        ref_transform = src.transform
        ref_crs = src.crs

    disp_cache: Dict[Tuple[datetime.datetime, datetime.datetime], np.ndarray] = {}
    for m_date, s_date, disp_path, _ in pairs:
        incidence_angle = read_incidence_angle(disp_path, DEFAULT_INCIDENCE_ANGLE)
        disp_cache[(m_date, s_date)] = load_disp_for_reference(
            disp_path, ref_shape, ref_transform, ref_crs, incidence_angle
        )
    return disp_cache


def filter_pairs_by_loop_closure(
    pairs: List[Tuple[datetime.datetime, datetime.datetime, str, Optional[str]]],
    threshold_mm: float,
    disp_cache: Optional[Dict[Tuple[datetime.datetime, datetime.datetime], np.ndarray]] = None,
) -> List[Tuple[datetime.datetime, datetime.datetime, str, Optional[str]]]:
    if len(pairs) < 3:
        return pairs

    if disp_cache is None:
        disp_cache = preload_disp_cache(pairs)

    pair_set = {(m_date, s_date) for m_date, s_date, _, _ in pairs}
    dates = sorted({d for pair in pair_set for d in pair})
    bad_pairs = set()
    threshold_m = threshold_mm / 1000.0
    for i, a in enumerate(dates):
        for j in range(i + 1, len(dates)):
            b = dates[j]
            for k in range(j + 1, len(dates)):
                c = dates[k]
                triplet = (a, b), (b, c), (a, c)
                if not all(pair in pair_set for pair in triplet):
                    continue
                disp_ab = disp_cache[(a, b)]
                disp_bc = disp_cache[(b, c)]
                disp_ac = disp_cache[(a, c)]
                closure = disp_ab + disp_bc - disp_ac
                valid = np.isfinite(closure)
                if not np.any(valid):
                    continue
                score = np.nanmedian(np.abs(closure[valid]))
                if score > threshold_m:
                    bad_pairs.update(triplet)

    filtered = [item for item in pairs if (item[0], item[1]) not in bad_pairs]
    return filtered if filtered else pairs


def preprocess_one(
    config: PipelineConfig, zip_file: str, subswath: str, s_burst: int, e_burst: int
) -> str:
    os.makedirs(config.temp_dir, exist_ok=True)
    validate_input_zip(zip_file)
    base = os.path.splitext(os.path.basename(zip_file))[0]
    out_dim = os.path.join(config.temp_dir, f"{base}_{subswath}_Split_Orb.dim")
    if os.path.exists(out_dim) and os.path.exists(out_dim.replace(".dim", ".data")):
        return out_dim

    xml_path = os.path.join(config.temp_dir, f"run_prep_{base}_{subswath}.xml")
    create_xml_file(
        XML_PREPROCESS,
        {
            "input": zip_file,
            "output": out_dim,
            "subswath": subswath,
            "startBurst": s_burst,
            "endBurst": e_burst,
        },
        xml_path,
    )

    run_gpt(config, xml_path, f"prep_{base}_{subswath}")
    return out_dim


def find_snaphu_work_dir(base: str) -> Optional[str]:
    for r, _, f in os.walk(base):
        if "snaphu.conf" in f:
            return r
    return None


def get_width_from_hdr(hdr: str) -> Optional[str]:
    if not os.path.exists(hdr):
        return None
    with open(hdr, "r", errors="ignore") as f:
        for l in f:
            if "samples" in l.lower() and "=" in l:
                return l.split("=")[1].strip()
    return None


def fix_snaphu_conf(conf: str, work_dir: str) -> None:
    with open(conf, "r", errors="ignore") as f:
        lines = f.readlines()

    coh = glob.glob(os.path.join(work_dir, "coh_*.img"))
    coh_name = os.path.basename(coh[0]) if coh else None

    new_lines = []
    has_init_method = False
    has_cost_thresh = False
    for l in lines:
        if l.strip().startswith("INITMETHOD"):
            new_lines.append("INITMETHOD MST\n")
            has_init_method = True
            continue
        if l.strip().startswith("COSTTHRESH"):
            new_lines.append("COSTTHRESH 600\n")
            has_cost_thresh = True
            continue
        if "NTILEROW" in l or "NTILECOL" in l:
            new_lines.append(l.split()[0] + " 1\n")
            continue
        if "ROWOVERLAP" in l or "COLOVERLAP" in l:
            new_lines.append(l.split()[0] + " 0\n")
            continue
        if l.strip().startswith("CORRFILE"):
            if coh_name:
                new_lines.append(f"CORRFILE {coh_name}\n")
            else:
                new_lines.append(f"# {l.strip()}\n")
            continue
        new_lines.append(l)

    if not has_init_method:
        new_lines.append("INITMETHOD MST\n")
    if not has_cost_thresh:
        new_lines.append("COSTTHRESH 600\n")

    with open(conf, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def run_pair(
    config: PipelineConfig,
    swath: str,
    m_date: datetime.datetime,
    s_date: datetime.datetime,
    m_dim: str,
    s_dim: str,
    wkt: str,
) -> Tuple[datetime.datetime, datetime.datetime, str, Optional[str]]:
    pair_name = f"{swath}_{m_date.strftime('%Y%m%d')}_{s_date.strftime('%Y%m%d')}"
    pair_dir = os.path.join(config.output_dir, pair_name)
    os.makedirs(pair_dir, exist_ok=True)
    status_path = os.path.join(pair_dir, "status.json")
    status = load_status(status_path)

    # Step1：Core IFG
    core_dim = os.path.join(pair_dir, "core_ifg.dim")
    if not os.path.exists(core_dim) or not status.get("core_ifg"):
        esd_node = ""
        pre_ifg = "Back-Geocoding"
        if config.use_esd:
            esd_node = r"""
  <node id="ESD"><operator>Enhanced-Spectral-Diversity</operator>
    <sources><sourceProduct refid="Back-Geocoding"/></sources>
  </node>"""
            pre_ifg = "ESD"

        xml_step1 = XML_STEP1_CORE.replace("${ESD_NODE}", esd_node).replace(
            "${PRE_IFG_NODE}", pre_ifg
        )

        xml_path = os.path.join(pair_dir, "run_core.xml")
        create_xml_file(
            xml_step1,
            {
                "master": m_dim,
                "slave": s_dim,
                "demName": config.dem_name,
                "output": core_dim,
            },
            xml_path,
        )
        run_gpt(config, xml_path, f"core_{pair_name}", work_dir=pair_dir)
        status["core_ifg"] = True
        save_status(status_path, status)

    # Step2：Filter + Subset + SnaphuExport
    subset_dim = os.path.join(pair_dir, "subset_ifg.dim")
    xml_path = os.path.join(pair_dir, "run_filter.xml")
    snaphu_dir = find_snaphu_work_dir(pair_dir)
    is_step2_done = os.path.exists(subset_dim) and (snaphu_dir is not None)

    if not is_step2_done or not status.get("subset_ifg"):
        create_xml_file(
            XML_STEP2_FILTER,
            {
                "input": core_dim,
                "demName": config.dem_name,
                "wkt": wkt,
                "targetFolder": pair_dir,
                "output": subset_dim,
            },
            xml_path,
        )
        run_gpt(config, xml_path, f"filter_{pair_name}", work_dir=pair_dir)
        status["subset_ifg"] = True
        save_status(status_path, status)

    snaphu_dir = find_snaphu_work_dir(pair_dir)
    if not snaphu_dir:
        raise RuntimeError("SnaphuExport 未生成 snaphu.conf（可能空数据/不重叠/DEM失败）")

    # Step3：snaphu unwrap
    phase_imgs = glob.glob(os.path.join(snaphu_dir, "Phase_*.img"))
    if not phase_imgs:
        raise RuntimeError("snaphu_dir 下找不到 Phase_*.img")

    phase_img = phase_imgs[0]
    phase_name = os.path.basename(phase_img)
    unw_name = "UnwPhase_" + phase_name.replace("Phase_", "")
    deep_unw = os.path.join(snaphu_dir, unw_name)
    out_unw = os.path.join(pair_dir, unw_name)

    if not os.path.exists(out_unw) or not status.get("snaphu"):
        conf = os.path.join(snaphu_dir, "snaphu.conf")
        fix_snaphu_conf(conf, snaphu_dir)
        width = get_width_from_hdr(phase_img.replace(".img", ".hdr"))
        if not width:
            raise RuntimeError("无法从 .hdr 读取 width(samples=)")

        subprocess.check_call(
            [config.snaphu_cmd, "-f", "snaphu.conf", phase_name, width, "-o", unw_name],
            cwd=snaphu_dir,
        )

        shutil.move(deep_unw, out_unw)
        hdr_src = deep_unw.replace(".img", ".hdr")
        hdr_dst = out_unw.replace(".img", ".hdr")
        if os.path.exists(hdr_src):
            shutil.move(hdr_src, hdr_dst)
        status["snaphu"] = True
        save_status(status_path, status)

    # Step4：Export displacement & coherence
    disp_tif = os.path.join(config.output_dir, f"Result_{pair_name}_disp.tif")
    if not os.path.exists(disp_tif) or not status.get("disp_export"):
        xml_path = os.path.join(pair_dir, "run_export_disp.xml")
        create_xml_file(
            XML_GEO_DISP,
            {
                "wrapped": subset_dim,
                "unwrapped": out_unw.replace(".img", ".hdr"),
                "demName": config.dem_name,
                "outFile": disp_tif,
            },
            xml_path,
        )
        run_gpt(config, xml_path, f"disp_{pair_name}", work_dir=pair_dir)
        status["disp_export"] = True
        save_status(status_path, status)

    # coherence band（失败则降级不掩膜）
    pair_date_str = f"{m_date.strftime('%d%b%Y')}_{s_date.strftime('%d%b%Y')}"
    coh_band = f"coh_{swath}_VV_{pair_date_str}"
    coh_tif = os.path.join(config.output_dir, f"Result_{pair_name}_coh.tif")
    if not os.path.exists(coh_tif) or not status.get("coh_export"):
        xml_path = os.path.join(pair_dir, "run_export_coh.xml")
        create_xml_file(
            XML_GEO_COH,
            {
                "input": subset_dim,
                "demName": config.dem_name,
                "cohBand": coh_band,
                "outFile": coh_tif,
            },
            xml_path,
        )
        try:
            run_gpt(config, xml_path, f"coh_{pair_name}", work_dir=pair_dir)
            status["coh_export"] = True
            save_status(status_path, status)
        except Exception:
            coh_tif = None

    return m_date, s_date, disp_tif, coh_tif


def calc_cumulative(
    config: PipelineConfig,
    tif_pairs: List[Tuple[datetime.datetime, datetime.datetime, str, Optional[str]]],
    out_name: str,
) -> Tuple[Optional[str], Optional[str], Dict[str, float], Dict[str, float]]:
    if not tif_pairs:
        return None, None, {}, {}

    with rasterio.open(tif_pairs[0][2]) as src0:
        ref_meta = src0.meta.copy()
        ref_crs = src0.crs
        ref_trans = src0.transform
        h, w = src0.height, src0.width

    dem = None
    if config.topo_phase_correction and os.path.exists(config.dem_tif_path):
        with rasterio.open(config.dem_tif_path) as dem_src:
            dem = np.full((h, w), np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(dem_src, 1),
                destination=dem,
                src_transform=dem_src.transform,
                src_crs=dem_src.crs,
                dst_transform=ref_trans,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )
    elif config.topo_phase_correction:
        LOGGER.warning("未找到 DEM_TIF_PATH，跳过线性高程修正")

    dates = sorted({d for pair in tif_pairs for d in pair[:2]})
    date_index = {date: idx for idx, date in enumerate(dates)}
    n_dates = len(dates)
    n_pairs = len(tif_pairs)

    disp_stack: List[np.ndarray] = []
    weight_stack: List[np.ndarray] = []
    coherence_stack: List[np.ndarray] = []
    baseline_vector = np.zeros(n_pairs, dtype=np.float32)

    for idx, (m_date, s_date, disp_path, coh_path) in enumerate(tif_pairs):
        incidence_angle = read_incidence_angle(disp_path, config.incidence_angle)
        angle_cos = cos(radians(incidence_angle))
        if angle_cos == 0:
            angle_cos = cos(radians(config.incidence_angle))

        with rasterio.open(disp_path) as src:
            disp = np.full((h, w), np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=disp,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_trans,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )
        disp = disp / angle_cos

        coherence = None
        if coh_path and os.path.exists(coh_path):
            with rasterio.open(coh_path) as src:
                coh = np.zeros((h, w), dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=coh,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_trans,
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear,
                )
            coherence = coh
            coherence_stack.append(coh)

        mask = build_reliability_mask(
            disp,
            coherence,
            config.coherence_threshold,
            config.jump_gradient_threshold,
        )

        if dem is not None:
            disp = linear_topo_phase_correction(
                disp, dem, mask, config.topo_phase_correction
            )

        disp = remove_planar_ramp(disp, mask, config.orbital_ramp_removal)
        disp = apply_atmospheric_filter(
            disp, mask, config.atm_filter_block_size
        )

        weights = np.ones((h, w), dtype=np.float32)
        if config.weighted_stacking and coherence is not None:
            weights = np.square(np.clip(coherence, 0.0, 1.0))
        weights[~mask] = 0.0
        disp[~mask] = np.nan

        disp_stack.append(disp)
        weight_stack.append(weights)
        baseline_vector[idx] = read_perpendicular_baseline(disp_path)

    reference = select_reference_point(
        coherence_stack, dem, config.reference_coherence_percentile
    )
    if reference:
        r, c = reference
        for i in range(n_pairs):
            ref_val = disp_stack[i][r, c]
            if np.isfinite(ref_val):
                disp_stack[i] = disp_stack[i] - ref_val

    disp_cube = np.stack(disp_stack, axis=0)
    weight_cube = np.stack(weight_stack, axis=0)

    A = np.zeros((n_pairs, n_dates), dtype=np.float32)
    for i, (m_date, s_date, _, _) in enumerate(tif_pairs):
        mi = date_index[m_date]
        si = date_index[s_date]
        if mi > 0:
            A[i, mi] = -1.0
        if si > 0:
            A[i, si] = 1.0
    A = np.concatenate([A[:, 1:], baseline_vector[:, None]], axis=1)

    times = np.array([(d - dates[0]).days / 365.25 for d in dates], dtype=np.float32)
    times = times[1:]

    rate = np.full((h, w), np.nan, dtype=np.float32)
    cumulative_disp = np.full((h, w), np.nan, dtype=np.float32)
    for row in range(h):
        for col in range(w):
            L = disp_cube[:, row, col]
            W = weight_cube[:, row, col]
            valid = np.isfinite(L) & (W > 0)
            if np.count_nonzero(valid) < max(3, A.shape[1]):
                continue
            A_valid = A[valid]
            L_valid = L[valid]
            W_sqrt = np.sqrt(W[valid])
            Aw = A_valid * W_sqrt[:, None]
            Lw = L_valid * W_sqrt
            try:
                x, _, _, _ = np.linalg.lstsq(Aw, Lw, rcond=None)
            except np.linalg.LinAlgError:
                continue
            displacements = x[:-1]
            if displacements.size == 0:
                continue
            cumulative_disp[row, col] = displacements[-1]
            if displacements.size != times.size:
                continue
            if config.robust_estimation:
                median = np.nanmedian(displacements)
                mad = np.nanmedian(np.abs(displacements - median))
                if mad > 0:
                    mask = np.abs(displacements - median) <= (
                        config.robust_mad_threshold * mad
                    )
                    if np.any(mask):
                        displacements = displacements[mask]
                        t_sel = times[mask]
                    else:
                        t_sel = times
                else:
                    t_sel = times
            else:
                t_sel = times
            if displacements.size < 2:
                continue
            A_rate = np.vstack([t_sel, np.ones_like(t_sel)]).T
            rate_params, _, _, _ = np.linalg.lstsq(A_rate, displacements, rcond=None)
            rate[row, col] = rate_params[0]

    if config.output_mm:
        rate = rate * 1000.0
        cumulative_disp = cumulative_disp * 1000.0
        out_name = out_name.replace(".tif", "_mm_per_year.tif")

    nodata = -9999.0
    out = np.where(np.isnan(rate), nodata, rate).astype(np.float32)
    cum_out = np.where(np.isnan(cumulative_disp), nodata, cumulative_disp).astype(
        np.float32
    )

    ref_meta.update(dtype=rasterio.float32, count=1, nodata=nodata, compress="deflate")
    out_path = os.path.join(config.output_dir, out_name)
    with rasterio.open(out_path, "w", **ref_meta) as dst:
        dst.write(out, 1)
    cumulative_name = (
        "Final_Combined_Cumulative_Subsidence_mm.tif"
        if config.output_mm
        else "Final_Combined_Cumulative_Subsidence.tif"
    )
    cumulative_path = os.path.join(config.output_dir, cumulative_name)
    with rasterio.open(cumulative_path, "w", **ref_meta) as dst:
        dst.write(cum_out, 1)
    return out_path, cumulative_path, compute_stats(rate), compute_stats(cumulative_disp)


def mosaic_results(
    config: PipelineConfig, tif_paths: List[str], out_name: str
) -> Tuple[Optional[str], Dict[str, float]]:
    if not tif_paths:
        return None, {}
    srcs = [rasterio.open(p) for p in tif_paths]
    mosaic, out_trans = merge(srcs)
    out_meta = srcs[0].meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "compress": "deflate",
        }
    )
    out_path = os.path.join(config.output_dir, out_name)
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(mosaic)
    for s in srcs:
        s.close()
    stats = compute_stats(mosaic[0]) if mosaic.size > 0 else {}
    return out_path, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="D-InSAR pipeline")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--shp-path", default=DEFAULT_SHP_PATH)
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--gpt-path", default=DEFAULT_GPT_PATH)
    parser.add_argument("--snaphu-cmd", default=DEFAULT_SNAPHU_CMD)
    parser.add_argument("--dem-tif-path", default=DEFAULT_DEM_TIF_PATH)
    parser.add_argument("--run-id", default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_CPU_TASKS)
    parser.add_argument(
        "--max-temporal-baseline",
        type=int,
        default=DEFAULT_MAX_TEMPORAL_BASELINE_DAYS,
        help="SBAS 网络最大时间基线（天）",
    )
    parser.add_argument(
        "--required-disk-gb",
        type=int,
        default=DEFAULT_REQUIRED_DISK_GB,
        help="磁盘预检所需最小剩余空间（GB）",
    )
    parser.add_argument(
        "--loop-closure-threshold-mm",
        type=float,
        default=DEFAULT_LOOP_CLOSURE_THRESHOLD_MM,
        help="闭合差阈值（毫米）",
    )
    parser.add_argument(
        "--atm-filter-block-size",
        type=int,
        default=DEFAULT_ATM_FILTER_BLOCK_SIZE,
        help="大气滤波空间块大小（像素）",
    )
    parser.add_argument(
        "--jump-gradient-threshold",
        type=float,
        default=DEFAULT_JUMP_GRADIENT_THRESHOLD,
        help="解缠跳变梯度阈值（位移幅度）",
    )
    parser.add_argument(
        "--no-robust-estimation",
        action="store_false",
        dest="robust_estimation",
        default=DEFAULT_ROBUST_ESTIMATION,
        help="禁用稳健估计（MAD 剔除）",
    )
    parser.add_argument(
        "--robust-mad-threshold",
        type=float,
        default=DEFAULT_ROBUST_MAD_THRESHOLD,
        help="MAD 异常剔除倍数",
    )
    parser.add_argument(
        "--reference-coherence-percentile",
        type=float,
        default=DEFAULT_REFERENCE_COHERENCE_PERCENTILE,
        help="参考点选择的平均相干性分位数",
    )
    args = parser.parse_args()

    output_base_dir = os.path.join(args.project_root, "Output")
    output_dir = os.path.join(output_base_dir, args.run_id)
    temp_dir = os.path.join(output_dir, "Temp_Preprocessed")
    jvm_heap, tile_cache, adjusted_max_workers = compute_memory_settings(
        args.max_workers
    )
    tile_cache = normalize_tile_cache(jvm_heap, tile_cache)
    config = PipelineConfig(
        input_dir=args.input_dir,
        shp_path=args.shp_path,
        project_root=args.project_root,
        output_base_dir=output_base_dir,
        output_dir=output_dir,
        temp_dir=temp_dir,
        gpt_path=args.gpt_path,
        snaphu_cmd=args.snaphu_cmd,
        max_cpu_tasks=adjusted_max_workers,
        threads_per_worker=DEFAULT_THREADS_PER_WORKER,
        jvm_heap=jvm_heap,
        tile_cache=tile_cache,
        incidence_angle=DEFAULT_INCIDENCE_ANGLE,
        coherence_threshold=DEFAULT_COHERENCE_THRESHOLD,
        weighted_stacking=DEFAULT_WEIGHTED_STACKING,
        orbital_ramp_removal=DEFAULT_ORBITAL_RAMP_REMOVAL,
        topo_phase_correction=DEFAULT_TOPO_PHASE_CORRECTION,
        dem_tif_path=args.dem_tif_path,
        dem_name=DEFAULT_DEM_NAME,
        use_esd=DEFAULT_USE_ESD,
        output_mm=DEFAULT_OUTPUT_MM,
        required_disk_gb=args.required_disk_gb,
        max_temporal_baseline_days=args.max_temporal_baseline,
        loop_closure_threshold_mm=args.loop_closure_threshold_mm,
        atm_filter_block_size=args.atm_filter_block_size,
        jump_gradient_threshold=args.jump_gradient_threshold,
        robust_estimation=args.robust_estimation,
        robust_mad_threshold=args.robust_mad_threshold,
        reference_coherence_percentile=args.reference_coherence_percentile,
    )

    setup_logging(os.path.join(config.output_dir, "logs"), args.run_id)
    if adjusted_max_workers != args.max_workers:
        LOGGER.info("调整 max_workers: %s -> %s", args.max_workers, adjusted_max_workers)
    LOGGER.info("JVM_HEAP=%s TILE_CACHE=%s", jvm_heap, tile_cache)
    validate_environment(config)
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.temp_dir, exist_ok=True)

    zips = sorted(glob.glob(os.path.join(config.input_dir, "*.zip")))
    dated_files = []
    for z in zips:
        date = get_date_from_zip(z)
        if date is not None:
            dated_files.append((date, z))
    file_map = sorted(dated_files, key=lambda x: x[0])
    if len(file_map) < 2:
        LOGGER.error("影像不足 2 景")
        return

    LOGGER.info("[Phase 0] 分析覆盖条带 & Burst...")
    tasks, wkt = analyze_all_subswaths(file_map[0][1], config.shp_path)
    if not tasks:
        LOGGER.error("SHP 与影像无交集或元数据读取失败")
        return

    swath_final_tifs: List[str] = []
    quality_report: Dict[str, Any] = {"run_id": args.run_id, "swaths": {}, "final": {}}

    for task in tasks:
        swath = task["swath"]
        s_burst, e_burst = task["s"], task["e"]
        LOGGER.info("=== 条带 %s Burst %s-%s ===", swath, s_burst, e_burst)

        pre_map = {}
        for d, z in file_map:
            LOGGER.info(
                "预处理: %s | %s | Burst %s-%s",
                d.strftime("%Y%m%d"),
                swath,
                s_burst,
                e_burst,
            )
            pre_map[d] = preprocess_one(config, z, swath, s_burst, e_burst)

        pairs = build_sbas_pairs(file_map, config.max_temporal_baseline_days)
        if not pairs:
            LOGGER.warning("未生成任何配对，请检查 max_temporal_baseline 设置")
            continue
        tif_pairs: List[Tuple[str, Optional[str]]] = []
        tif_pairs_with_dates: List[
            Tuple[datetime.datetime, datetime.datetime, str, Optional[str]]
        ] = []

        LOGGER.info("[Phase 2] 干涉计算并行（max_workers=%s）...", config.max_cpu_tasks)
        with ProcessPoolExecutor(max_workers=config.max_cpu_tasks) as ex:
            futs = []
            for m, s in pairs:
                futs.append(
                    ex.submit(run_pair, config, swath, m, s, pre_map[m], pre_map[s], wkt)
                )

            for f in as_completed(futs):
                try:
                    m_date, s_date, disp_tif, coh_tif = f.result()
                    tif_pairs_with_dates.append((m_date, s_date, disp_tif, coh_tif))
                    LOGGER.info("完成: %s", os.path.basename(disp_tif))
                except Exception as e:
                    LOGGER.exception("pair 失败: %s", e)

        disp_cache = preload_disp_cache(tif_pairs_with_dates)
        filtered_pairs = filter_pairs_by_loop_closure(
            tif_pairs_with_dates, config.loop_closure_threshold_mm, disp_cache
        )
        if len(filtered_pairs) < len(tif_pairs_with_dates):
            LOGGER.warning(
                "闭合差剔除 %s/%s 个干涉对",
                len(tif_pairs_with_dates) - len(filtered_pairs),
                len(tif_pairs_with_dates),
            )
        swath_out, cumulative_out, stats, cumulative_stats = calc_cumulative(
            config, filtered_pairs, f"Total_Subsidence_{swath}.tif"
        )
        if swath_out:
            LOGGER.info("条带累计完成: %s", swath_out)
            swath_final_tifs.append(swath_out)
            quality_report["swaths"][swath] = {
                "path": swath_out,
                "stats": stats,
                "cumulative_path": cumulative_out,
                "cumulative_stats": cumulative_stats,
            }

    final, final_stats = mosaic_results(
        config, swath_final_tifs, "Final_Combined_Subsidence_Vertical_Masked.tif"
    )
    if final:
        LOGGER.info("最终拼接完成: %s", final)
        quality_report["final"] = {"path": final, "stats": final_stats}
    else:
        LOGGER.warning("未生成最终结果")

    report_path = os.path.join(config.output_dir, "quality_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
