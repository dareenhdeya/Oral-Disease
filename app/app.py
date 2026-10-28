import json
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

model = tf.keras.models.load_model(
    "resnet_final.keras",
    custom_objects={"preprocess_input": preprocess_input},
    safe_mode=False,
)
thresholds = json.load(open("thresholds.json"))
classes = list(thresholds.keys())


def predict(image):
    if image is None:
        return "<div class='msg'><div class='msg-icon'>🦷</div>Upload a photo first.</div>"

    img = Image.fromarray(image).convert("RGB").resize((224, 224))
    x = np.array(img, dtype="float32")[None, ...]
    probs = model.predict(x, verbose=0)[0]

    # every disease above its own threshold, not just the top one
    found = [(c, p) for c, p in zip(classes, probs) if p >= thresholds[c]]
    found.sort(key=lambda t: -t[1])

    if found:
        chips = ""
        for i, (c, p) in enumerate(found):
            chips += (
                f"<div class='chip' style='animation-delay:{i * 0.15}s'>"
                f"<span class='dot'></span>{c}<b>{p * 100:.0f}%</b></div>"
            )
        head = (
            f"<div class='status found-status'><span class='status-icon'>⚠️</span>"
            f"<h3>{len(found)} condition{'s' if len(found) > 1 else ''} detected</h3></div>"
            f"<div class='chips'>{chips}</div>"
        )
    else:
        head = (
            "<div class='status clear-status'><span class='status-icon'>✅</span>"
            "<h3>No condition detected</h3></div>"
        )

    rows = ""
    ranked = sorted(zip(classes, probs), key=lambda t: -t[1])
    for i, (c, p) in enumerate(ranked):
        hit = "hit" if p >= thresholds[c] else ""
        rows += (
            f"<div class='row' style='animation-delay:{i * 0.08}s'>"
            f"<div class='label'>{c}<span>{p * 100:.0f}%</span></div>"
            f"<div class='track'>"
            f"<div class='fill {hit}' style='--w:{p * 100:.1f}%; animation-delay:{0.2 + i * 0.1}s'></div>"
            f"<div class='mark' style='left:{thresholds[c] * 100:.1f}%'></div>"
            f"</div></div>"
        )

    note = (
        "<p class='note'>Each condition has its own detection threshold (the tick on each bar), "
        "so a detected condition can show a low score.</p>"
    )
    return f"<div class='result'>{head}<div class='bars'>{rows}</div>{note}</div>"


EMPTY = "<div class='msg'><div class='msg-icon'>🦷</div>Upload a photo and press Analyze.</div>"


def run(image, analyzed):
    # Clear mode: reset everything
    if analyzed:
        return None, EMPTY, False, gr.update(value="Analyze")

    if image is None:
        return (
            image,
            "<div class='msg'><div class='msg-icon'>🦷</div>Upload a photo first.</div>",
            False,
            gr.update(value="Analyze"),
        )

    html = predict(image)
    return image, html, True, gr.update(value="Clear")

CSS = """
/* ---------- animated background ---------- */
.gradio-container {
  background: linear-gradient(135deg, #0b1020, #1a1040, #0b1a30, #1a1040) !important;
  background-size: 400% 400% !important;
  animation: bgshift 18s ease infinite;
  min-height: 100vh;
  position: relative;
  overflow-x: hidden;
}
@keyframes bgshift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }

/* floating glowing orbs */
.gradio-container::before, .gradio-container::after {
  content:""; position:fixed; border-radius:50%; filter:blur(90px); opacity:.35;
  z-index:0; pointer-events:none;
}
.gradio-container::before {
  width:420px; height:420px; background:#8b5cf6; top:-100px; left:-100px;
  animation: float1 14s ease-in-out infinite;
}
.gradio-container::after {
  width:380px; height:380px; background:#06b6d4; bottom:-100px; right:-80px;
  animation: float2 16s ease-in-out infinite;
}
@keyframes float1 { 0%,100%{transform:translate(0,0)} 50%{transform:translate(120px,80px)} }
@keyframes float2 { 0%,100%{transform:translate(0,0)} 50%{transform:translate(-120px,-70px)} }

.gradio-container > * { position:relative; z-index:1; }

/* ---------- hero (smaller) ---------- */
.hero {
  text-align:center; padding:14px 18px; border-radius:18px; margin-bottom:6px;
  background:rgba(255,255,255,.05);
  border:1px solid rgba(255,255,255,.12);
  backdrop-filter:blur(14px);
  box-shadow:0 0 30px rgba(139,92,246,.25);
  animation: dropin .8s cubic-bezier(.2,.9,.3,1.2);
}
@keyframes dropin { from{opacity:0; transform:translateY(-30px) scale(.96)} to{opacity:1; transform:none} }
.hero .tooth { font-size:2rem; display:inline-block; animation: bob 2.6s ease-in-out infinite; }
@keyframes bob { 0%,100%{transform:translateY(0) rotate(-6deg)} 50%{transform:translateY(-7px) rotate(6deg)} }
.hero h1 {
  margin:2px 0 4px; font-size:1.6rem; font-weight:800;
  background:linear-gradient(90deg,#a78bfa,#22d3ee,#f0abfc,#a78bfa);
  background-size:300% 100%;
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
  animation: shine 6s linear infinite;
}
@keyframes shine { to{background-position:300% 0} }
.hero p { margin:0; color:#cbd5e1; font-size:.9rem; }

/* ---------- glass panels ---------- */
.glass {
  background:rgba(255,255,255,.05) !important;
  border:1px solid rgba(255,255,255,.12) !important;
  border-radius:20px !important;
  backdrop-filter:blur(14px);
  padding:16px !important;
  animation: rise .7s ease-out both;
}
.glass:hover { box-shadow:0 0 30px rgba(34,211,238,.18); transition:box-shadow .4s; }

/* ---------- image upload box (no more black) ---------- */
.upload-box,
.upload-box > div,
.upload-box .image-container,
.upload-box .upload-container,
.upload-box button,
.upload-box [data-testid="image"] {
  background: transparent !important;
}
.upload-box {
  border: 2px dashed rgba(34,211,238,.45) !important;
  border-radius: 16px !important;
  background: linear-gradient(135deg, rgba(139,92,246,.10), rgba(6,182,212,.10)) !important;
  animation: borderpulse 3s ease-in-out infinite;
  transition: box-shadow .3s, transform .3s;
  overflow: hidden;
  position: relative;
}
.upload-box:hover {
  box-shadow: 0 0 28px rgba(34,211,238,.45), inset 0 0 22px rgba(139,92,246,.2);
  transform: translateY(-2px);
}
@keyframes borderpulse {
  0%,100%{ border-color: rgba(34,211,238,.35) }
  50%    { border-color: rgba(167,139,250,.85) }
}
/* floating upload icon */
.upload-box svg { animation: bob 2.6s ease-in-out infinite; }
/* label chip and bottom toolbar */
.upload-box .label-wrap, .upload-box [data-testid="block-label"] {
  background: rgba(139,92,246,.35) !important; color:#fff !important;
  border-color: rgba(255,255,255,.15) !important;
}
.upload-box .image-frame, .upload-box .icon-buttons, .upload-box .source-selection {
  background: rgba(255,255,255,.04) !important;
  border-color: rgba(255,255,255,.1) !important;
}

/* scanning line on the image while processing */
.upload-box.generating::after, .upload-box.pending::after {
  content:""; position:absolute; left:0; right:0; height:3px; z-index:5;
  background:linear-gradient(90deg,transparent,#22d3ee,transparent);
  box-shadow:0 0 18px 6px rgba(34,211,238,.7);
  animation: scan 1.4s ease-in-out infinite;
}
@keyframes scan { 0%{top:0} 50%{top:calc(100% - 3px)} 100%{top:0} }

/* ---------- button ---------- */
.analyze-btn {
  background:linear-gradient(90deg,#8b5cf6,#06b6d4,#8b5cf6) !important;
  background-size:250% 100% !important;
  border:none !important; color:#fff !important; font-weight:700 !important;
  font-size:1.05rem !important; border-radius:14px !important; padding:14px !important;
  animation: btnflow 5s linear infinite, glow 2.4s ease-in-out infinite;
  transition: transform .2s;
}
.analyze-btn:hover { transform:translateY(-3px) scale(1.02); }
.analyze-btn:active { transform:scale(.97); }
@keyframes btnflow { to{background-position:250% 0} }
@keyframes glow {
  0%,100%{box-shadow:0 0 14px rgba(139,92,246,.5)}
  50%{box-shadow:0 0 28px rgba(6,182,212,.8)} }

/* ---------- results ---------- */
.result { animation: rise .6s ease-out; }
@keyframes rise { from{opacity:0; transform:translateY(18px)} to{opacity:1; transform:none} }

.status { display:flex; align-items:center; gap:12px; margin-bottom:12px; }
.status h3 { margin:0; font-size:1.3rem; }
.status-icon { font-size:1.8rem; animation: wiggle 1.8s ease-in-out infinite; }
@keyframes wiggle { 0%,100%{transform:rotate(0)} 25%{transform:rotate(-12deg)} 75%{transform:rotate(12deg)} }
.found-status h3 { color:#fb7185; }
.clear-status h3 { color:#34d399; }

.chips { display:flex; flex-wrap:wrap; gap:10px; margin-bottom:22px; }
.chip {
  display:flex; align-items:center; gap:8px; padding:9px 16px; border-radius:999px;
  background:rgba(251,113,133,.12); color:#fecdd3; border:1px solid rgba(251,113,133,.5);
  font-weight:600; opacity:0; animation: pop .5s cubic-bezier(.2,.9,.3,1.4) forwards;
}
.chip b { background:linear-gradient(90deg,#f43f5e,#fb7185); color:#fff; border-radius:999px; padding:2px 9px; font-size:.8rem; }
.dot { width:9px; height:9px; border-radius:50%; background:#fb7185; animation: pulse 1.6s infinite; }
@keyframes pop { from{opacity:0; transform:scale(.6) rotate(-6deg)} to{opacity:1; transform:scale(1) rotate(0)} }
@keyframes pulse {
  0%{box-shadow:0 0 0 0 rgba(251,113,133,.7)}
  70%{box-shadow:0 0 0 9px rgba(251,113,133,0)}
  100%{box-shadow:0 0 0 0 rgba(251,113,133,0)} }

/* ---------- probability bars ---------- */
.row { margin-bottom:14px; opacity:0; animation: slidein .5s ease-out forwards; }
@keyframes slidein { from{opacity:0; transform:translateX(-20px)} to{opacity:1; transform:none} }
.label { display:flex; justify-content:space-between; font-size:.92rem; margin-bottom:5px; color:#e2e8f0; }
.label span { color:#22d3ee; font-weight:700; }
.track { position:relative; height:13px; border-radius:9px; background:rgba(255,255,255,.08); overflow:visible; }
.fill {
  height:100%; width:0; border-radius:9px;
  background:linear-gradient(90deg,#475569,#94a3b8);
  animation: grow 1s cubic-bezier(.2,.8,.2,1) forwards;
}
.fill.hit {
  background:linear-gradient(90deg,#8b5cf6,#ec4899,#f43f5e,#ec4899,#8b5cf6);
  background-size:300% 100%;
  box-shadow:0 0 10px rgba(236,72,153,.35);
  animation: grow 1s cubic-bezier(.2,.8,.2,1) forwards, hue 12s ease-in-out infinite;
}
@keyframes hue {
  0%, 100% { background-position: 0% 0; }
  50%      { background-position: 100% 0; }
}
@keyframes grow { from{width:0} to{width:var(--w)} }

.mark { position:absolute; top:-4px; width:2px; height:21px; background:#fff; opacity:.7; border-radius:2px; }

.note { font-size:.8rem; color:#94a3b8; margin-top:16px; line-height:1.5; }

/* empty state */
.msg { padding:40px 20px; text-align:center; color:#94a3b8; }
.msg-icon { font-size:3rem; margin-bottom:8px; animation: bob 2.6s ease-in-out infinite; }

/* general text colors for dark mode */
.gradio-container label, .gradio-container .prose, .gradio-container span { color:#e2e8f0; }


/* ---------- equal height boxes + slightly smaller ---------- */
.main-row { align-items: stretch !important; }
.main-row > .glass {
  display: flex !important;
  flex-direction: column !important;
  align-self: stretch !important;
  padding: 12px !important;
}
.result-col > div, .result-col .prose, .result-col .html-container {
  flex: 1 1 auto;
}

.hero { padding: 30px 32px !important; }
.hero .tooth { font-size: 2.3rem !important; }
.hero h1 { font-size: 1.8rem !important; margin: 2px 0 4px !important; }
.hero p { font-size: .95rem !important; }
.status h3 { font-size: 1.2rem !important; }
.status-icon { font-size: 1.6rem !important; }
.chips { margin-bottom: 14px !important; gap: 8px !important; }
.chip { padding: 6px 13px !important; font-size: .95rem !important; }
.row { margin-bottom: 10px !important; }
.label { font-size: .95rem !important; margin-bottom: 3px !important; }
.label span { font-size: 1rem !important; }
.track { height: 10px !important; }
.mark { top: -3px !important; height: 16px !important; }
.note { font-size: .7rem !important; margin-top: 10px !important; }
.analyze-btn { padding: 10px !important; font-size: .98rem !important; }
.msg { padding: 28px 16px !important; }

/* hide the clear (X) and fullscreen buttons on the image */
.upload-box button[aria-label="Clear"],
.upload-box button[aria-label="Remove Image"],
.upload-box button[aria-label="Fullscreen"],
.upload-box button[aria-label="Exit fullscreen"],
.upload-box .icon-button-wrapper.top-panel {
  display: none !important;
}

"""
with gr.Blocks() as demo:
    gr.HTML(
        "<div class='hero'><div class='tooth'>🦷</div>"
        "<h1>Oral Disease Detection</h1>"
        "<p>Upload a photo of the teeth or mouth and the model will check for common conditions.</p></div>"
    )

    with gr.Row(elem_classes="main-row", equal_height=True):
        with gr.Column(scale=4, min_width=300, elem_classes="glass"):
            image = gr.Image(label="Photo", show_label=False, height=344, elem_classes="upload-box")
            button = gr.Button("Analyze", variant="primary", elem_classes="analyze-btn")
        with gr.Column(scale=6, min_width=300, elem_classes=["glass", "result-col"]):
            output = gr.HTML(
                "<div class='msg'><div class='msg-icon'>🦷</div>Upload a photo and press Analyze.</div>"
            )

    analyzed = gr.State(False)

    button.click(
        run,
        inputs=[image, analyzed],
        outputs=[image, output, analyzed, button],
    )

    def on_image_change(img):
        if img is None:
            return gr.update(value="Analyze"), False, EMPTY
        return gr.update(value="Analyze"), False, gr.update()

    image.upload(
        lambda: (gr.update(value="Analyze"), False, EMPTY),
        inputs=None,
        outputs=[button, analyzed, output],
    )

demo.launch(css=CSS, theme=gr.themes.Base(primary_hue="violet", secondary_hue="cyan"))