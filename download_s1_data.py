import argparse
import logging
import os
import time
import zipfile
from dataclasses import dataclass
from typing import Iterable, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_URLS = [
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20170301T222800_20170301T222835_004521_007DF6_0538.zip",
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20170406T222801_20170406T222836_005046_008D38_4707.zip",
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20170512T222803_20170512T222837_005571_009C20_71AD.zip",
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20170605T222804_20170605T222839_005921_00A630_0FD5.zip",
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20170711T222806_20170711T222841_006446_00B559_C2FC.zip",
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20170804T222808_20170804T222842_006796_00BF5D_66EB.zip",
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20170909T222809_20170909T222844_007321_00CEA1_A843.zip",
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20171003T222810_20171003T222845_007671_00D8C5_59A6.zip",
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20171108T222810_20171108T222844_008196_00E7CC_E5A9.zip",
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20171202T222810_20171202T222844_008546_00F2A1_0301.zip",
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20180107T222808_20180107T222843_009071_01037C_D3C6.zip",
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20180212T222807_20180212T222842_009596_0114B5_4E65.zip",
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20180308T222807_20180308T222841_009946_012052_B62E.zip",
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20180401T222808_20180401T222842_010296_012BB1_111F.zip",
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20180507T222809_20180507T222843_010821_013C94_DE6E.zip",
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20180612T222811_20180612T222846_011346_014D58_A961.zip",
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20180718T222813_20180718T222848_011871_015D9C_7BE8.zip",
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20180823T222815_20180823T222850_012396_016DAC_F535.zip",
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20180916T222816_20180916T222851_012746_017879_2D3D.zip",
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20181010T222817_20181010T222852_013096_018329_62DA.zip",
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20181115T222817_20181115T222851_013621_019386_72FA.zip",
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20181221T222816_20181221T222850_014146_01A48F_5B80.zip",
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20190114T222815_20190114T222849_014496_01AFFD_2755.zip",
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20190219T222814_20190219T222848_015021_01C117_9864.zip",
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20190315T222814_20190315T222848_015371_01CC80_B5BF.zip",
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20190420T222815_20190420T222849_015896_01DDCD_8A0C.zip",
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20190514T222816_20190514T222850_016246_01E939_DC55.zip",
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20190619T222818_20190619T222852_016771_01F909_1ED7.zip",
    "https://datapool.asf.alaska.edu/SLC/SB/S1B_IW_SLC__1SDV_20190713T222819_20190713T222854_017121_02035E_9110.zip",
]


@dataclass(frozen=True)
class DownloadConfig:
    save_dir: str
    username: str
    password: str
    timeout_s: int
    chunk_size_mb: int
    min_valid_size_gb: float
    max_retries: int
    backoff_factor: float


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def build_session(config: DownloadConfig) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=config.max_retries,
        backoff_factor=config.backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.auth = (config.username, config.password)
    session.get("https://urs.earthdata.nasa.gov")
    return session


def load_urls(urls: Optional[List[str]], url_file: Optional[str]) -> List[str]:
    result: List[str] = []
    if urls:
        result.extend(urls)
    if url_file:
        with open(url_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    result.append(line)
    return result or DEFAULT_URLS


def format_size(num_bytes: float) -> str:
    gb = num_bytes / (1024**3)
    return f"{gb:.2f} GB"


def download_file(url: str, session: requests.Session, config: DownloadConfig) -> None:
    filename = url.split("/")[-1]
    save_path = os.path.join(config.save_dir, filename)
    temp_path = f"{save_path}.part"

    min_valid_size = config.min_valid_size_gb * 1024**3
    if os.path.exists(save_path) and os.path.getsize(save_path) >= min_valid_size:
        logging.info("已存在，跳过: %s", filename)
        return

    existing_path = temp_path if os.path.exists(temp_path) else save_path
    existing_size = os.path.getsize(existing_path) if os.path.exists(existing_path) else 0
    if 0 < existing_size < min_valid_size and existing_path == save_path:
        logging.warning("发现疑似损坏文件，删除重下: %s", save_path)
        os.remove(save_path)
        existing_size = 0
        existing_path = temp_path

    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        logging.info("续传: %s (%s)", filename, format_size(existing_size))
    else:
        logging.info("下载: %s", filename)

    with session.get(url, stream=True, timeout=config.timeout_s, headers=headers) as resp:
        if resp.status_code == 401 or "login" in resp.url:
            logging.info("进行 Earthdata 认证重定向: %s", filename)
            resp = session.get(resp.url, stream=True, timeout=config.timeout_s)
        resp.raise_for_status()

        if existing_size > 0 and resp.status_code != 206:
            logging.warning("服务器不支持续传，重新下载: %s", filename)
            existing_size = 0
            existing_path = temp_path

        mode = "ab" if resp.status_code == 206 else "wb"
        total_size = int(resp.headers.get("content-length", 0)) + existing_size
        chunk_size = config.chunk_size_mb * 1024 * 1024

        downloaded = existing_size
        start_time = time.time()
        last_log_time = start_time
        with open(existing_path, mode) as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if total_size > 0 and now - last_log_time >= 5:
                    percent = downloaded / total_size * 100
                    speed = downloaded / max(now - start_time, 1e-6) / 1024**2
                    logging.info(
                        "进度: %s %s/%s (%.1f%%, %.1f MB/s)",
                        filename,
                        format_size(downloaded),
                        format_size(total_size),
                        percent,
                        speed,
                    )
                    last_log_time = now

    if os.path.exists(existing_path) and existing_path != save_path:
        os.replace(existing_path, save_path)
    try:
        with zipfile.ZipFile(save_path) as zip_file:
            bad_member = zip_file.testzip()
            if bad_member:
                raise zipfile.BadZipFile(f"坏文件成员: {bad_member}")
    except zipfile.BadZipFile as exc:
        logging.warning("ZIP 校验失败，删除并标记失败: %s (%s)", filename, exc)
        os.remove(save_path)
        raise

    logging.info("完成: %s", filename)


def download_all(urls: Iterable[str], config: DownloadConfig) -> None:
    os.makedirs(config.save_dir, exist_ok=True)
    session = build_session(config)
    for url in urls:
        try:
            download_file(url, session, config)
        except Exception as exc:
            logging.exception("下载失败: %s (%s)", url, exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sentinel-1 SLC 批量下载脚本")
    parser.add_argument("--username", default=os.getenv("EARTHDATA_USERNAME", ""))
    parser.add_argument("--password", default=os.getenv("EARTHDATA_PASSWORD", ""))
    parser.add_argument("--save-dir", default=r"D:\leixiang\D-InSAR\S1_Data")
    parser.add_argument("--url", action="append", dest="urls")
    parser.add_argument("--url-file")
    parser.add_argument("--timeout-s", type=int, default=60)
    parser.add_argument("--chunk-size-mb", type=int, default=4)
    parser.add_argument("--min-valid-size-gb", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--backoff-factor", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    if not args.username or not args.password:
        raise ValueError("请通过 --username/--password 或环境变量提供 Earthdata 账号密码")

    config = DownloadConfig(
        save_dir=args.save_dir,
        username=args.username,
        password=args.password,
        timeout_s=args.timeout_s,
        chunk_size_mb=args.chunk_size_mb,
        min_valid_size_gb=args.min_valid_size_gb,
        max_retries=args.max_retries,
        backoff_factor=args.backoff_factor,
    )
    urls = load_urls(args.urls, args.url_file)
    download_all(urls, config)


if __name__ == "__main__":
    main()
