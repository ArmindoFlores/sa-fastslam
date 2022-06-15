import argparse
import os
import tempfile
import zipfile

import colorama
import requests

SERVER = "sigma03.ist.utl.pt"
PORT = 53764


def main(ns):
    print(ns)
    with tempfile.TemporaryDirectory() as temp_dir:
        print("[+] Compressing scripts...")
        with zipfile.ZipFile(os.path.join(temp_dir, "scripts.zip"), "w") as zf:
            for file in filter(lambda f: f.endswith(".py"), os.listdir()):
                print(f"[+] Writing {file}...")
                zf.write(file)
            print(f"{colorama.Fore.GREEN}[+] Done!{colorama.Style.RESET_ALL}")
        print("[+] Sending to server...")
        with open(os.path.join(temp_dir, "scripts.zip"), "rb") as zf:
            try:
                query = f"email={ns.email}"
                if ns.tag is not None:
                    query += f"&tag={ns.tag}"
                r = requests.post(f"http://{SERVER}:{PORT}/run?{query}", files={"scripts.zip": zf})
            except requests.ConnectionError:
                print(f"{colorama.Fore.RED}[!] Error: server is unreachable{colorama.Style.RESET_ALL}")
                return
        if r.status_code != 200:
            print(f"{colorama.Fore.RED}[!] Error: Invalid server response!{colorama.Style.RESET_ALL}")
            return
    print(f"{colorama.Fore.GREEN}[+] Done!{colorama.Style.RESET_ALL}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FastSLAM simulation remotely")
    parser.add_argument("email", type=str, help="e-mail of the recipient")
    parser.add_argument("--tag", type=str, help="tag associated with this simulation")
    ns = parser.parse_args()
    main(ns)
