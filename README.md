# D-InSAR Pipeline

基于 ESA SNAP GPT 与 snaphu 的 D-InSAR 处理流水线脚本，支持：

- Sentinel-1 SLC 数据自动条带/子波束分析与裁剪
- 干涉、滤波、解缠、地形校正与位移输出
- 多景相邻对叠加与最终马赛克拼接
- 可选相干性加权、平面坡度移除与线性高程修正

## 运行环境

- Python 3.8+
- ESA SNAP（含 `gpt` 命令行）
- snaphu
- 依赖库：`geopandas`, `shapely`, `numpy`, `rasterio`

## 快速开始

1. 准备 Sentinel-1 SLC `.zip` 数据放入 `INPUT_DIR` 目录。
2. 准备目标区域 `.shp` 文件。
3. 确认 SNAP GPT 与 snaphu 可执行路径正确。

执行：

```bash
python dinsar_pipeline.py
```

## 常用参数

```bash
python dinsar_pipeline.py \
  --input-dir "D:\leixiang\D-InSAR\S1_Data" \
  --shp-path "D:\leixiang\D-InSAR\GreatWall_Buffer\Hebei_Baoding_1km.shp" \
  --project-root "D:\leixiang\D-InSAR" \
  --gpt-path "C:\Program Files\esa-snap\bin\gpt.exe" \
  --snaphu-cmd "D:\leixiang\D-InSAR\Software\snaphu-v1.4.2_win64\bin\snaphu.exe" \
  --dem-tif-path "D:\leixiang\D-InSAR\DEM\dem.tif" \
  --max-workers 2
```

## 输出结构

默认输出目录为 `${project_root}/Output/${run_id}`，包括：

- `Temp_Preprocessed/`：预处理缓存
- `logs/`：运行日志
- `Result_*_disp.tif`：相邻对位移结果
- `Result_*_coh.tif`：相邻对相干性结果
- `Total_Subsidence_<SWATH>.tif`：条带累计
- `Final_Combined_Subsidence_Vertical_Masked.tif`：最终拼接
- `quality_report.json`：质量统计报告

## 配置说明

脚本内部的默认参数位于 `dinsar_pipeline.py` 的 `DEFAULT_*` 常量中，运行时可通过命令行覆盖：

- `DEFAULT_MAX_CPU_TASKS`：并行 pair 数量
- `DEFAULT_THREADS_PER_WORKER`：GPT 线程数
- `DEFAULT_JVM_HEAP` / `DEFAULT_TILE_CACHE`：Java 资源
- `DEFAULT_COHERENCE_THRESHOLD`：相干性阈值
- `DEFAULT_WEIGHTED_STACKING`：相干性加权堆叠开关
- `DEFAULT_ORBITAL_RAMP_REMOVAL`：平面坡度移除开关
- `DEFAULT_TOPO_PHASE_CORRECTION`：线性高程修正开关

## 运行建议

- 建议先在小区域与少量影像上测试参数与流程。
- 初次运行可将 `DEFAULT_MAX_CPU_TASKS` 设置为 1 以降低资源压力。
- snaphu 解缠依赖相干性，若解缠失败需检查数据质量与配置。
