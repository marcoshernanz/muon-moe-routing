"""Poll Vast.ai for a host of four RTX 5090s fit for the second chain, and exit when one is found at or under a price.

Fit: verified and rentable, 200 GB of disk, a driver of 580 or newer (the lockfile's torch is built for CUDA 13), cards whose power can reach the given watts (575 by default; a 500 W card was 3.5 percent slower, EVIDENCE-LOG section 13), a reliability of 0.95 or more, not in China (the Hub and Docker Hub go through proxies there), and not host 124072 (Japan), which refused the SSH key twice on 2026-09-23, nor host 410852, whose US machine could not reach Docker Hub on 2026-09-25. The price is the hour's total with 200 GB of disk. Every look is appended to outputs/offers.log. Usage: offers.py <max dollars an hour> <minutes to poll> [seconds between looks] [watts]; exits 0 printing the fit offers when the cheapest is at or under the price, and 1 printing the best fit offers when the time runs out.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LOG = Path(__file__).parent / "outputs" / "offers.log"
QUERY = "num_gpus=4 gpu_name=RTX_5090 verified=true rentable=true disk_space>=200"
DISK_GB = 200
SKIP_HOSTS = {124072, 410852}


def fit(watts: float = 575) -> list[dict]:
    raw = subprocess.run(["vastai", "search", "offers", QUERY, "-o", "dph_total", "--raw"], capture_output=True, text=True, timeout=120).stdout
    offers = []
    for x in json.loads(raw):
        driver = int(str(x.get("driver_version") or "0").split(".")[0])
        if driver < 580 or (x.get("gpu_max_power") or 0) < watts or (x.get("reliability2") or 0) < 0.95 or x.get("host_id") in SKIP_HOSTS or str(x.get("geolocation")).endswith("CN"):
            continue
        total = x["dph_base"] + x["storage_cost"] * DISK_GB / 730
        offers.append({"id": x["id"], "host": x["host_id"], "machine": x["machine_id"], "dollars_h": round(total, 3), "where": x.get("geolocation"),
                       "driver": x.get("driver_version"), "max_w": x.get("gpu_max_power"), "reliability": round(x.get("reliability2") or 0, 3),
                       "down_mbps": round(x.get("inet_down") or 0), "cpu": x.get("cpu_name"), "threads": x.get("cpu_cores_effective")})
    return sorted(offers, key=lambda o: o["dollars_h"])


def main() -> None:
    price, minutes = float(sys.argv[1]), float(sys.argv[2])
    every = float(sys.argv[3]) if len(sys.argv) > 3 else 180
    watts = float(sys.argv[4]) if len(sys.argv) > 4 else 575
    end = time.time() + 60 * minutes
    LOG.parent.mkdir(exist_ok=True)
    while True:
        try:
            offers = fit(watts)
        except Exception as error:  # a failed search is one look lost, not the poll
            offers = []
            print(f"{datetime.now():%F %T} search failed: {error}", file=sys.stderr)
        with LOG.open("a") as log:
            log.write(f"{datetime.now():%F %T} {len(offers)} fit: " + json.dumps(offers[:5]) + "\n")
        if offers and offers[0]["dollars_h"] <= price:
            print("FOUND"); print("\n".join(json.dumps(o) for o in offers[:5])); sys.exit(0)
        if time.time() > end:
            print("TIMEOUT"); print("\n".join(json.dumps(o) for o in offers[:5])); sys.exit(1)
        time.sleep(every)


if __name__ == "__main__":
    main()
