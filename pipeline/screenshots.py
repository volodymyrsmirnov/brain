"""Render the README screenshots and the link-preview image from the live viewer.

Usage (needs Playwright, which is not in requirements.txt):
    uv venv work/pwenv && VIRTUAL_ENV=work/pwenv uv pip install playwright && work/pwenv/bin/playwright install chromium
    work/pwenv/bin/python pipeline/screenshots.py            # writes docs/img/*.jpg and docs/img/og.jpg

It serves docs/ on a free local port, opens each guided view with the menu and captions hidden,
and saves a JPEG. The views and camera tweaks are listed in SHOTS below.
"""
import functools
import http.server
import socketserver
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
DOCS, OUT = ROOT / "docs", ROOT / "docs" / "img"

# name, guided view id, extra wait (ms), distance factor (1.0 = fit the visible meshes), optional JS filter of what to frame
SHOTS = [
    ("volume", "volume", 1500, 0.62),
    ("glass-head", "glass", 1500, 1.0),
    ("skull", "skull", 1500, 1.0),
    ("eyes", "eyes", 1500, 1.0),
    ("brain-map", "atlas", 1500, 1.0),
    ("explode", "parts", 5000, 1.0),
    ("inflated", "inflate", 7500, 1.0),
    ("split", "split", 1500, 0.8),
    ("deep", "deep", 1500, 1.0),
    ("hypothalamus", "hypothalamus", 1500, 1.25),
    ("cerebellum", "cerebellum", 1500, 1.0),
    ("inner-ear", "ears", 1500, 1.0, "it.L.hemi === 'R'"),
    ("veins", "veins", 1500, 1.0),
    ("slice", "slice", 2500, 1.0),
]

HIDE_UI = """
for (const id of ['panel', 'caption', 'toggle', 'gh', 'mesh-load']) { const e = document.getElementById(id); if (e) e.style.display = 'none'; }
document.querySelectorAll('.modal').forEach(m => m.classList.remove('show'));
dispatchEvent(new Event('resize'));
"""


def serve():
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a): pass
    handler = functools.partial(Quiet, directory=str(DOCS))
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def open_viewer(browser, port, w, h):
    page = browser.new_page(viewport={"width": w, "height": h})
    page.goto(f"http://127.0.0.1:{port}/")
    page.wait_for_function("window.viewer && window.viewer.meshesReady", timeout=120_000)
    page.evaluate("async () => { await window.viewer.meshesReady; }")
    page.evaluate(HIDE_UI)
    return page


def show(page, tour, wait_ms, zoom, focus="true"):
    page.evaluate(f"window.viewer.startTour('{tour}')")
    page.wait_for_timeout(1200 + wait_ms)
    page.evaluate(f"""() => {{
        const v = window.viewer, T = v.THREE; v.controls.autoRotate = false;
        // Re-frame on the visible meshes, keeping the view direction (volume-only views keep their camera).
        const vis = v.items.filter(it => it.mesh && it.mesh.visible && it.mat.opacity > 0.3 && ({focus}));
        let target = v.controls.target.clone(), dist = v.camera.position.distanceTo(target);
        if (vis.length) {{
            const box = new T.Box3(); vis.forEach(it => box.expandByObject(it.mesh));
            target = box.getCenter(new T.Vector3());
            dist = box.getSize(new T.Vector3()).length() * 1.55;
        }}
        const dir = v.camera.position.clone().sub(v.controls.target).normalize();
        v.camera.position.copy(target).addScaledVector(dir, dist * {zoom}); v.controls.target.copy(target); v.controls.update();
        document.getElementById('caption').style.display = 'none';
    }}""")
    page.wait_for_timeout(700)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    httpd, port = serve()
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"])
        page = open_viewer(browser, port, 1200, 800)
        for name, tour, *opts in SHOTS:
            show(page, tour, *opts)
            page.screenshot(path=str(OUT / f"{name}.jpg"), type="jpeg", quality=86)
            print("saved", name)
        page.close()

        # Link preview (Open Graph / Twitter card): 1200x630 with a title on the left.
        page = open_viewer(browser, port, 1200, 630)
        show(page, "glass", 1500, 0.85)
        page.evaluate("""() => {
            const d = document.createElement('div');
            d.style.cssText = 'position:fixed;left:56px;top:50%;transform:translateY(-50%);max-width:430px;z-index:50;font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#e8eaf0';
            d.innerHTML = '<div style="font-size:54px;font-weight:800;line-height:1.05">My head,<br>in 3D</div>'
              + '<div style="font-size:24px;color:#aab1bd;margin-top:18px;line-height:1.35">An interactive model built from my own brain MRI with open-source neuroscience tools</div>'
              + '<div style="font-size:20px;color:#7aa2ff;margin-top:22px">volodymyrsmirnov.github.io/brain</div>';
            document.body.appendChild(d);
            const v = window.viewer;   // shift the head to the right half of the card
            v.camera.setViewOffset(1200, 630, -260, 0, 1200, 630); v.camera.updateProjectionMatrix();
        }""")
        page.wait_for_timeout(600)
        page.screenshot(path=str(OUT / "og.jpg"), type="jpeg", quality=88)
        print("saved og")
        browser.close()
    httpd.shutdown()


if __name__ == "__main__":
    main()
