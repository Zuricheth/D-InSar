import os
import glob
import subprocess
import datetime
import zipfile
import re
import shutil
import logging
import xml.etree.ElementTree as ET

import geopandas as gpd
from shapely.geometry import Point

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.merge import merge

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, List, Tuple, Dict

# ================= 服务器配置（按你给的） =================
INPUT_DIR = r"D:\leixiang\D-InSAR\S1_Data"
SHP_PATH = r"D:\leixiang\D-InSAR\GreatWall_Buffer\Hebei_Baoding_1km.shp"
PROJECT_ROOT = r"D:\leixiang\D-InSAR"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Output")
TEMP_DIR = os.path.join(OUTPUT_DIR, "Temp_Preprocessed")

GPT_PATH = r"C:\Program Files\esa-snap\bin\gpt.exe"
SNAPHU_CMD = r"D:\leixiang\D-InSAR\Software\snaphu-v1.4.2_win64\bin\snaphu.exe"

MAX_CPU_TASKS = 2  # 并行 gpt/snaphu 的进程数（先跑通可改 1）
THREADS_PER_WORKER = 40  # 每个 GPT 进程线程
JVM_HEAP = "32G"  # 每个 GPT 进程堆
TILE_CACHE = "12G"

INCIDENCE_ANGLE = 39.5
COHERENCE_THRESHOLD = 0.30

# 精度向选项
DEM_NAME = "SRTM 1Sec HGT"  # 想和本地一致就改回 "SRTM 3Sec"
USE_ESD = True  # TOPS 配准增强
OUTPUT_MM = True  # 输出毫米（PhaseToDisplacement 通常是米）
# =========================================================


LOGGER = logging.getLogger(__name__)


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
      <alpha>1.0</alpha>
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
      <statCostMode>DEFO</statCostMode>
      <initMethod>MCF</initMethod>
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


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def validate_environment() -> None:
    missing = []
    for path in [INPUT_DIR, SHP_PATH, GPT_PATH, SNAPHU_CMD]:
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        missing_str = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(f"缺少必要路径/可执行文件:\n{missing_str}")


def create_xml_file(template: str, replace_dict: Dict[str, str], out_path: str) -> None:
    content = template
    for k, v in replace_dict.items():
        content = content.replace("${" + k + "}", str(v).replace("\\", "/"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)


def run_gpt(xml_path: str, task_id: str, work_dir: Optional[str] = None) -> None:
    os.makedirs(TEMP_DIR, exist_ok=True)
    local_tmp = os.path.join(TEMP_DIR, f"java_tmp_{task_id}")
    os.makedirs(local_tmp, exist_ok=True)

    java_tmp = local_tmp.replace("\\", "/")
    env = os.environ.copy()
    env["JAVA_TOOL_OPTIONS"] = f"-Xmx{JVM_HEAP} -Djava.io.tmpdir={java_tmp} -XX:+UseG1GC"

    log_path = os.path.join(TEMP_DIR, f"gpt_{task_id}.log")
    cmd = [GPT_PATH, xml_path, "-q", str(THREADS_PER_WORKER), "-c", TILE_CACHE]

    with open(log_path, "w", encoding="utf-8") as log:
        p = subprocess.run(
            cmd,
            cwd=work_dir,
            env=env,
            stdout=log,
            stderr=log,
            check=False,
        )

    shutil.rmtree(local_tmp, ignore_errors=True)

    if p.returncode != 0:
        raise RuntimeError(f"GPT 失败({task_id})，查看日志: {log_path}")


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


def preprocess_one(zip_file: str, subswath: str, s_burst: int, e_burst: int) -> str:
    os.makedirs(TEMP_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(zip_file))[0]
    out_dim = os.path.join(TEMP_DIR, f"{base}_{subswath}_Split_Orb.dim")
    if os.path.exists(out_dim) and os.path.exists(out_dim.replace(".dim", ".data")):
        return out_dim

    xml_path = os.path.join(TEMP_DIR, f"run_prep_{base}_{subswath}.xml")
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

    run_gpt(xml_path, f"prep_{base}_{subswath}")
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
    for l in lines:
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

    with open(conf, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def run_pair(
    swath: str,
    m_date: datetime.datetime,
    s_date: datetime.datetime,
    m_dim: str,
    s_dim: str,
    wkt: str,
) -> Tuple[str, Optional[str]]:
    pair_name = f"{swath}_{m_date.strftime('%Y%m%d')}_{s_date.strftime('%Y%m%d')}"
    pair_dir = os.path.join(OUTPUT_DIR, pair_name)
    os.makedirs(pair_dir, exist_ok=True)

    # Step1：Core IFG
    core_dim = os.path.join(pair_dir, "core_ifg.dim")
    if not os.path.exists(core_dim):
        esd_node = ""
        pre_ifg = "Back-Geocoding"
        if USE_ESD:
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
                "demName": DEM_NAME,
                "output": core_dim,
            },
            xml_path,
        )
        run_gpt(xml_path, f"core_{pair_name}", work_dir=pair_dir)

    # Step2：Filter + Subset + SnaphuExport
    subset_dim = os.path.join(pair_dir, "subset_ifg.dim")
    xml_path = os.path.join(pair_dir, "run_filter.xml")
    snaphu_dir = find_snaphu_work_dir(pair_dir)
    is_step2_done = os.path.exists(subset_dim) and (snaphu_dir is not None)

    if not is_step2_done:
        create_xml_file(
            XML_STEP2_FILTER,
            {
                "input": core_dim,
                "demName": DEM_NAME,
                "wkt": wkt,
                "targetFolder": pair_dir,
                "output": subset_dim,
            },
            xml_path,
        )
        run_gpt(xml_path, f"filter_{pair_name}", work_dir=pair_dir)

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

    if not os.path.exists(out_unw):
        conf = os.path.join(snaphu_dir, "snaphu.conf")
        fix_snaphu_conf(conf, snaphu_dir)
        width = get_width_from_hdr(phase_img.replace(".img", ".hdr"))
        if not width:
            raise RuntimeError("无法从 .hdr 读取 width(samples=)")

        subprocess.check_call(
            [SNAPHU_CMD, "-f", "snaphu.conf", phase_name, width, "-o", unw_name],
            cwd=snaphu_dir,
        )

        shutil.move(deep_unw, out_unw)
        hdr_src = deep_unw.replace(".img", ".hdr")
        hdr_dst = out_unw.replace(".img", ".hdr")
        if os.path.exists(hdr_src):
            shutil.move(hdr_src, hdr_dst)

    # Step4：Export displacement & coherence
    disp_tif = os.path.join(OUTPUT_DIR, f"Result_{pair_name}_disp.tif")
    if not os.path.exists(disp_tif):
        xml_path = os.path.join(pair_dir, "run_export_disp.xml")
        create_xml_file(
            XML_GEO_DISP,
            {
                "wrapped": subset_dim,
                "unwrapped": out_unw.replace(".img", ".hdr"),
                "demName": DEM_NAME,
                "outFile": disp_tif,
            },
            xml_path,
        )
        run_gpt(xml_path, f"disp_{pair_name}", work_dir=pair_dir)

    # coherence band（失败则降级不掩膜）
    pair_date_str = f"{m_date.strftime('%d%b%Y')}_{s_date.strftime('%d%b%Y')}"
    coh_band = f"coh_{swath}_VV_{pair_date_str}"
    coh_tif = os.path.join(OUTPUT_DIR, f"Result_{pair_name}_coh.tif")
    if not os.path.exists(coh_tif):
        xml_path = os.path.join(pair_dir, "run_export_coh.xml")
        create_xml_file(
            XML_GEO_COH,
            {
                "input": subset_dim,
                "demName": DEM_NAME,
                "cohBand": coh_band,
                "outFile": coh_tif,
            },
            xml_path,
        )
        try:
            run_gpt(xml_path, f"coh_{pair_name}", work_dir=pair_dir)
        except Exception:
            coh_tif = None

    return disp_tif, coh_tif


def calc_cumulative(
    tif_pairs: List[Tuple[str, Optional[str]]], out_name: str
) -> Optional[str]:
    if not tif_pairs:
        return None

    with rasterio.open(tif_pairs[0][0]) as src0:
        ref_meta = src0.meta.copy()
        ref_crs = src0.crs
        ref_trans = src0.transform
        h, w = src0.height, src0.width

    cum = np.full((h, w), np.nan, dtype=np.float32)

    for disp_path, coh_path in tif_pairs:
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

        mask = np.ones((h, w), dtype=bool)
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
            mask = coh >= COHERENCE_THRESHOLD

        disp[~mask] = np.nan
        cum = np.where(np.isnan(cum), disp, cum + disp)

    vert = cum / np.cos(np.deg2rad(INCIDENCE_ANGLE))

    nodata = -9999.0
    if OUTPUT_MM:
        vert = vert * 1000.0
        out_name = out_name.replace(".tif", "_mm.tif")

    out = np.where(np.isnan(vert), nodata, vert).astype(np.float32)

    ref_meta.update(dtype=rasterio.float32, count=1, nodata=nodata, compress="deflate")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    with rasterio.open(out_path, "w", **ref_meta) as dst:
        dst.write(out, 1)
    return out_path


def mosaic_results(tif_paths: List[str], out_name: str) -> Optional[str]:
    if not tif_paths:
        return None
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
    out_path = os.path.join(OUTPUT_DIR, out_name)
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(mosaic)
    for s in srcs:
        s.close()
    return out_path


def main() -> None:
    setup_logging()
    validate_environment()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    zips = sorted(glob.glob(os.path.join(INPUT_DIR, "*.zip")))
    dated_files = []
    for z in zips:
        date = get_date_from_zip(z)
        if date is not None:
            dated_files.append((date, z))
    file_map = sorted(dated_files, key=lambda x: x[0])
    if len(file_map) < 2:
        LOGGER.error("影像不足 2 景")
        return

    LOGGER.info("🔍 [Phase 0] 分析覆盖条带 & Burst...")
    tasks, wkt = analyze_all_subswaths(file_map[0][1], SHP_PATH)
    if not tasks:
        LOGGER.error("SHP 与影像无交集或元数据读取失败")
        return

    swath_final_tifs: List[str] = []

    for task in tasks:
        swath = task["swath"]
        s_burst, e_burst = task["s"], task["e"]
        LOGGER.info("🌊 === 条带 %s Burst %s-%s ===", swath, s_burst, e_burst)

        pre_map = {}
        for d, z in file_map:
            LOGGER.info(
                "⚙️ 预处理: %s | %s | Burst %s-%s",
                d.strftime("%Y%m%d"),
                swath,
                s_burst,
                e_burst,
            )
            pre_map[d] = preprocess_one(z, swath, s_burst, e_burst)

        pairs = [(file_map[i][0], file_map[i + 1][0]) for i in range(len(file_map) - 1)]
        tif_pairs: List[Tuple[str, Optional[str]]] = []

        LOGGER.info("🚀 [Phase 2] 干涉计算并行（max_workers=%s）...", MAX_CPU_TASKS)
        with ProcessPoolExecutor(max_workers=MAX_CPU_TASKS) as ex:
            futs = []
            for m, s in pairs:
                futs.append(ex.submit(run_pair, swath, m, s, pre_map[m], pre_map[s], wkt))

            for f in as_completed(futs):
                try:
                    disp_tif, coh_tif = f.result()
                    tif_pairs.append((disp_tif, coh_tif))
                    LOGGER.info("✅ 完成: %s", os.path.basename(disp_tif))
                except Exception as e:
                    LOGGER.exception("❌ pair 失败: %s", e)

        tif_pairs = sorted(tif_pairs, key=lambda x: x[0])
        swath_out = calc_cumulative(tif_pairs, f"Total_Subsidence_{swath}.tif")
        if swath_out:
            LOGGER.info("🎉 条带累计完成: %s", swath_out)
            swath_final_tifs.append(swath_out)

    final = mosaic_results(swath_final_tifs, "Final_Combined_Subsidence_Vertical_Masked.tif")
    if final:
        LOGGER.info("🏆 最终拼接完成: %s", final)
    else:
        LOGGER.warning("未生成最终结果")


if __name__ == "__main__":
    main()
