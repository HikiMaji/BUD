import requests
import time
import os

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
SHARE_URL = "https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V/folder/279976559690"
SHARED_NAME = "vxetzti0z0dv97yeh1jgxdprxxqucs2a"
# validate_chunks-20240808T030331Z-001.zip
TEST_FILE_ID = "1622190654556"

def test_proxy_connectivity(proxy_url):
    print(f"\n=================== Testing: {proxy_url} ===================")
    proxies = {"http": proxy_url, "https": proxy_url}
    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    session.proxies.update(proxies)
    
    # 1. Google 探测
    try:
        t0 = time.time()
        rg = session.get("https://www.google.com", timeout=5)
        print(f"[+] Google HTTP {rg.status_code} in {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"[-] Google failed: {e}")
        
    # 2. Box 探测
    try:
        t0 = time.time()
        print(f"[*] Connecting to Box...")
        r1 = session.get(SHARE_URL, timeout=8)
        print(f"[+] Box Share page HTTP {r1.status_code} in {time.time()-t0:.2f}s, cookies: {len(session.cookies)}")
        
        dl_url = f"https://ucla.app.box.com/index.php?rm=box_download_shared_file&shared_name={SHARED_NAME}&file_id=f_{TEST_FILE_ID}"
        t1 = time.time()
        r2 = session.get(dl_url, allow_redirects=False, timeout=8)
        loc = r2.headers.get("Location")
        if loc:
            print(f"[+] Got CDN direct link: {loc[:70]}...")
            t2 = time.time()
            r3 = session.get(loc, stream=True, timeout=8)
            total_bytes = 0
            chunk_t0 = time.time()
            for chunk in r3.iter_content(chunk_size=64*1024):
                total_bytes += len(chunk)
                if time.time() - chunk_t0 > 4:
                    break
            dur = time.time() - chunk_t0
            speed = (total_bytes / (1024*1024)) / dur if dur > 0 else 0
            print(f"[SUCCESS!] Box Speed: {speed:.2f} MB/s ({total_bytes/(1024*1024):.2f} MB in {dur:.2f}s)")
            return True
        else:
            print(f"[-] No Location: HTTP {r2.status_code}")
    except Exception as e:
        print(f"[-] Box failed: {e}")
    return False

if __name__ == "__main__":
    for p in ["http://127.0.0.1:7897", "http://127.0.0.1:10801", "http://127.0.0.1:17897"]:
        test_proxy_connectivity(p)
