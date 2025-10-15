import os, time, json, tempfile 
import numpy as np 
# live_emotion_pub.py
# nodo ros2 che ascolta dal microfono, stima (fallback se music2emo non disponibile),
# parla con llm e pubblica json d'azione su topic per unity

import os
import time
import json
import tempfile

import numpy as np
import pyaudio
import soundfile as sf

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# --- forza torch.load su CPU e con weights_only=False (PyTorch 2.6 safety change) ---
import torch
_orig_torch_load = torch.load
def _load_cpu_weights_only_false(f, *args, **kwargs):
    if "map_location" not in kwargs:
        kwargs["map_location"] = torch.device("cpu")
    # forziamo esplicitamente: il tuo ckpt è pickled completo, non solo state_dict
    kwargs["weights_only"] = False
    return _orig_torch_load(f, *args, **kwargs)
torch.load = _load_cpu_weights_only_false
# -----------------------------------------------------------------------------


# --- forza il path locale di Music2Emotion per l'import di music2emo.py ---
from pathlib import Path

# prova 1: variabile d'ambiente (se vuoi impostarla da shell)
m2e_dir = str(Path(__file__).resolve().parents[2] / "Music2Emotion")
if Path(m2e_dir).exists() and m2e_dir not in sys.path:
    sys.path.insert(0, m2e_dir)
# --------------------------------------------------------------------------


# tenta di importare music2emo; se non disponibile, userà un fallback neutro
try:
    from music2emo import Music2emo
    _m2e_available = True
except Exception as e:
    print("⚠️  music2emo import fallito, userò fallback neutro:", e)
    _m2e_available = False

# ===== config =====
chunk = 1024
fmt_rate = 16000
channels = 1
window_sec = 50
hop_sec = 5
change_threshold = 0.25
ema_alpha = 0.4
INTERACT_EVERY_SEC = 15
#DISLIKE_KEYWORDS = ["non voglio più", "ferma", "basta", "stop", "odio", "terribile", "schifo", "pessima", "brutta"]

# stato robot, inizializzato a riposo
robot_state = {"current_action": "rest", "last_action_ts": 0.0}


# llm (gemini)
import google.generativeai as genai
genai.configure(api_key = "APIKEY")
model = genai.GenerativeModel("gemini-1.5-pro-latest")


class ros2_bridge(Node):
    def __init__(self):
        super().__init__("music_emotion_pub")
        self.pub_action = self.create_publisher(String, "/music_emotion/action_json", 10)
        self.pub_state = self.create_publisher(String, "/music_emotion/state", 10)

    def publish_action(self, action_dict: dict):
        msg = String()
        msg.data = json.dumps(action_dict, ensure_ascii=False)
        self.pub_action.publish(msg)

    def publish_state(self, valence: float, arousal: float, tags: list[str]):
        payload = {"valence": float(valence), "arousal": float(arousal), "tags": tags}
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub_state.publish(msg)


def llm_inner_speech(valence, arousal, mood_tags):

    prompt = f"""
        Sei un robot sociale che ascolta musica in diretta.


        Segnali correnti (scala 1–9):
        - valence: {valence:.2f}
        - arousal: {arousal:.2f}
        - mood tags: {', '.join(mood_tags) if mood_tags else 'n/a'}


        Istruzioni di stile:
        - Rispondi in prima persona, in italiano, con tono empatico e conciso (max 2 frasi).
        - Varia il vocabolario; niente ripetizioni evidenti.
        - Termina con una breve domanda all'utente.
        - Non spiegare il ragionamento interno.
        """
    res = model.generate_content(prompt)
    return (res.text or "").strip()



def llm_decide_action(valence, arousal, mood_tags, user_reply):
    """
    LLM genera l'azione di alto livello dal solo contesto musicale + risposta utente.
    Output JSON minimale: {"action":"..."} (+ opzionali "gesture", "text")
    """
    generation_config = {
        "response_mime_type": "application/json",
        "response_schema": {
            "type": "object",
            "properties": {
                "action":  {"type": "string"},
                "gesture": {"type": "string"},
                "text":    {"type": "string"}
            },
            "required": ["action"]
        }
    }

    tags_str = ", ".join(mood_tags) if mood_tags else "n/a"

    prompt = f"""
Tu generi l'azione di un robot sociale che ascolta musica in diretta.

Segnali musicali (scala 1–9)
- valence: {valence:.2f}
- arousal: {arousal:.2f}
- mood tags: {tags_str}

Testo utente (linguaggio libero)
\"\"\"{user_reply}\"\"\"

Compito (unico step)
- Genera l’azione finale (non scegliere da lista fissa): deve emergere da emozione + intento utente.
- Consenso/sicurezza: se l’utente vuole interrompere o prova disagio (anche con parafrasi come “non voglio più sentire nulla”), genera {{ "action": "stop_music" }}.
- Se chiede pausa/riduzione → {{ "action": "rest" }}.
- Se chiede cambio brano/volume → {{ "action": "skip_track" }}, {{ "action": "lower_volume" }} o {{ "action": "raise_volume" }}.
- Altrimenti genera un’azione coerente con il contesto: es. "dance", "talk", "listen", "play".
- Se l’azione implica parlare, puoi includere un testo breve in "text".
- Produci SOLO JSON con "action" e, opzionali, "gesture" e "text".

Esempi (few-shot, non regole rigide)
- Utente: "Questa musica mi stressa, non voglio più sentirla" → {{ "action": "stop_music" }}
- Utente: "Puoi abbassare un po'?" → {{ "action": "lower_volume" }}
- Utente: "Cambiamo canzone?" → {{ "action": "skip_track" }}
- Utente: "Mi piace ma sono stanco" → {{ "action": "rest" }}
- Utente: "Che energia! andiamo!" → {{ "action": "dance", "gesture": "wave_arms" }}
- Utente: "Parliamone" → {{ "action": "talk", "text": "Certo, dimmi cosa provi ascoltando questa musica." }}

Rispondi SOLO con JSON conforme allo schema.
"""
    res = model.generate_content(prompt, generation_config=generation_config)
    return res.text



def analyze_audio_segment(audio_data: np.ndarray) -> dict:
    if not _m2e_available:
        return {"valence": 5.0, "arousal": 5.0, "predicted_moods": []}
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, audio_data, fmt_rate)
        try:
            out = Music2emo().predict(tmp_path)
        finally:
            os.remove(tmp_path)
        return out
    except Exception as e:
        print("⚠️  music2emo predict fallita, fallback neutro:", e)
        return {"valence": 5.0, "arousal": 5.0, "predicted_moods": []}


def _pick_input_device(pa: pyaudio.PyAudio) -> int | None:
    env = os.getenv("INPUT_DEVICE_INDEX")
    if env is not None and env.isdigit():
        return int(env)
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) > 0:
            return i
    return None



def run_live_session(node: ros2_bridge):
    pa = pyaudio.PyAudio()

    print("🔎 elenco dispositivi input disponibili:")
    for i in range(pa.get_device_count()):
        inf = pa.get_device_info_by_index(i)
        if inf.get("maxInputChannels", 0) > 0:
            rate = int(inf.get("defaultSampleRate", 0))
            print(f"  [{i}] {inf.get('name')} (rate max≈{rate} hz)")

    idx = _pick_input_device(pa)
    if idx is None:
        raise RuntimeError("nessun input device disponibile per pyaudio")
    print(f"🎚️ uso input_device_index={idx}")

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=fmt_rate,
        input=True,
        frames_per_buffer=chunk,
        input_device_index=idx
    )

    print("🎙️ listening live... ctrl+c per uscire")

    audio_buffer = np.array([], dtype=np.int16)
    smoothed = None
    last_interaction_time = 0.0  # cadenza delle domande

    try:
        while rclpy.ok():
            data = stream.read(chunk, exception_on_overflow=False)
            frame = np.frombuffer(data, dtype=np.int16)
            audio_buffer = np.append(audio_buffer, frame)

            ready = len(audio_buffer) >= window_sec * fmt_rate
            due   = (time.time() - last_interaction_time) >= INTERACT_EVERY_SEC

            if ready and due:
                # prepara finestra e stima
                segment = audio_buffer[-window_sec * fmt_rate:]
                audio_float = segment.astype(np.float32) / 32768.0
                out = analyze_audio_segment(audio_float)

                v, a = float(out["valence"]), float(out["arousal"])
                tags = out.get("predicted_moods", []) or []

                # smoothing
                cur = np.array([v, a], dtype=np.float32)
                smoothed = cur if smoothed is None else (ema_alpha * cur + (1 - ema_alpha) * smoothed)

                # 👉 pubblichiamo LO STATO solo ora (niente spam continuo)
                node.publish_state(float(smoothed[0]), float(smoothed[1]), tags)

                # frase empatica 
                text = llm_inner_speech(float(smoothed[0]), float(smoothed[1]), tags)
                print(f"\n🤖 robot: {text}")
                user_reply = input("🧑 you: ").strip()


                raw = llm_decide_action(float(smoothed[0]), float(smoothed[1]), tags, user_reply)
                try:
                    action_json = json.loads(raw) if raw else None
                except Exception:
                    action_json = None

                if not action_json or not isinstance(action_json.get("action"), str) or not action_json["action"]:
                    action_json = {"action": "talk"}

                print(json.dumps(action_json, indent=2, ensure_ascii=False))
                node.publish_action(action_json)


                robot_state["current_action"] = action_json.get("action", robot_state.get("current_action"))
                robot_state["last_action_ts"] = time.time()



                #if action_json.get("action") == "stop_music":
                 #   return



                # aggiorna cadenza
                last_interaction_time = time.time()

            rclpy.spin_once(node, timeout_sec=0.0)

    except KeyboardInterrupt:
        print("🛑 stop")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

def main():
    rclpy.init()
    node = ros2_bridge()
    try:
        run_live_session(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()