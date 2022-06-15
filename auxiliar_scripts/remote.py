import argparse
import os
import tempfile
import zipfile

import colorama
import requests

SERVER = "sigma02.ist.utl.pt"
PORT = 53764


def main(ns):
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
                r = requests.post(f"http://{SERVER}:{PORT}/run?email={ns.email}", files={"scripts.zip": zf})
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
    ns = parser.parse_args()
    main(ns)
