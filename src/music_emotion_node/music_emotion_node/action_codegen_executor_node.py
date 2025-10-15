# action_codegen_executor_node.py — LLM codegen sicura + esecuzione su NAO (qi.Session)
# Requisiti:
#   - ROS2 (rclpy)
#   - google-generativeai (Gemini) + GEMINI_API_KEY
#   - SDK NAOqi (modulo 'qi') disponibile sulla macchina
#
# Sottoscrive:
#   /music_emotion/state       (std_msgs/String)  -> {"valence":..,"arousal":..,"tags":[...]}
#   /music_emotion/action_json (std_msgs/String)  -> {"action":"...","gesture":"...","text":"..."}
#
# Flusso:
#   1) Riceve action_json dall'LLM (es. {"action":"stop_music"}).
#   2) Chiede a Gemini uno SCRIPT di movimento in DSL JSON (op whitelist).
#   3) Valida/limita lo script (range, max passi, sleep totale).
#   4) Esegue sul NAO via qi.Session (ALTextToSpeech, ALMotion, ALRobotPosture).
#
# Parametri ROS2:
#   - ip (string, default "192.168.0.103")
#   - port (int,    default 9559)
#
# Env opzionali:
#   - DRY_RUN=1  → non si connette al NAO, stampa cosa farebbe.

import os, sys, json, time
from typing import Any, Dict, List

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ---------- DRY RUN ----------
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"

# ---------- qi (NAOqi nuovo stile) ----------
try:
    import qi
except Exception as e:
    if not DRY_RUN:
        print(f"[FATAL] Modulo 'qi' non importabile: {e}")
        sys.exit(1)
    qi = None  # in dry-run possiamo procedere

# ---------- LLM (Gemini) ----------
import google.generativeai as genai
genai.configure(api_key = "APIKEY")
LLM_MODEL = genai.GenerativeModel("gemini-1.5-pro-latest")

def _loads_json(x: str) -> Dict[str, Any]:
    try:
        return json.loads(x)
    except Exception:
        return {}

# ----------------- Guard-rails della DSL -----------------
ALLOWED_OPS = {"say", "posture", "move_rel", "set_angles", "open_hand", "close_hand", "sleep"}

MAX_STEPS = 32
MAX_TOTAL_SLEEP_SEC = 12.0
ANGLE_MIN, ANGLE_MAX = -2.0, 2.0
SPEED_MIN, SPEED_MAX = 0.1, 1.0
MOVE_X_MIN, MOVE_X_MAX = -0.5, 0.5
MOVE_Y_MIN, MOVE_Y_MAX = -0.3, 0.3
MOVE_TH_MIN, MOVE_TH_MAX = -1.0, 1.0
SLEEP_MIN, SLEEP_MAX = 0.0, 3.0

HAND_MAP = {"R": "RHand", "L": "LHand", "r": "RHand", "l": "LHand"}

# ----------------- Nodo -----------------
class ActionCodegenExecutorQi(Node):
    def __init__(self):
        super().__init__("action_codegen_executor_qi")

        # --- parametri ROS2 (come NaoTTSNode) ---
        self.declare_parameter('ip', '192.168.0.101')
        self.declare_parameter('port', 9559)
        ip   = self.get_parameter('ip').get_parameter_value().string_value
        port = self.get_parameter('port').get_parameter_value().integer_value

        # --- stato emozionale ricevuto ---
        self.last_valence = 5.0
        self.last_arousal = 5.0
        self.last_tags: List[str] = []

        # --- connessione qi.Session & servizi ---
        self.session = None
        self.motion_service = None
        self.posture_service = None
        self.tts = None
        self.life_service = None
        self.awareness_service = None

        if DRY_RUN:
            self.get_logger().warn("DRY_RUN=1 → non mi collego al NAO; simulo le azioni.")
        else:
            self.session = self._connect_nao(ip, port)
            self.motion_service    = self.session.service("ALMotion")
            self.posture_service   = self.session.service("ALRobotPosture")
            self.tts               = self.session.service("ALTextToSpeech")
            # opzionali se presenti:
            try: self.life_service     = self.session.service("ALAutonomousLife")
            except Exception: pass
            try: self.awareness_service= self.session.service("ALBasicAwareness")
            except Exception: pass

            try: self.tts.setLanguage("Italian")
            except Exception: pass
            try: self.motion_service.wakeUp()
            except Exception: pass

        # --- subscriber ROS2 ---
        self.sub_state  = self.create_subscription(String, "/music_emotion/state", self._on_state, 10)
        self.sub_action = self.create_subscription(String, "/music_emotion/action_json", self._on_action, 10)

        self.get_logger().info(f"[codegen] pronto. ip={ip} port={port} dry={DRY_RUN}")

    # ---------- connessione ----------
    def _connect_nao(self, ip: str, port: int):
        sess = qi.Session()
        url = f"tcp://{ip}:{port}"
        try:
            sess.connect(url)
            self.get_logger().info(f"Connesso a {url}")
            return sess
        except RuntimeError as e:
            self.get_logger().fatal(f"Connessione fallita a {url}: {e}")
            raise

    # ---------- ROS callbacks ----------
    def _on_state(self, msg: String):
        data = _loads_json(msg.data)
        try:
            self.last_valence = float(data.get("valence", self.last_valence))
            self.last_arousal = float(data.get("arousal", self.last_arousal))
            tags = data.get("tags", [])
            self.last_tags = [str(t) for t in tags][:8] if isinstance(tags, list) else []
        except Exception:
            pass

    def _on_action(self, msg: String):
        payload = _loads_json(msg.data)
        action  = (payload.get("action")  or "").strip().lower()
        gesture = (payload.get("gesture") or "").strip()
        text    = (payload.get("text")    or "").strip()

        if not action:
            self.get_logger().warn("Azione vuota (niente codegen).")
            return

        self.get_logger().info(f"LLM action='{action}' gesture='{gesture or '-'}'  → codegen script")

        # 1) prompt LLM → script JSON
        script_json = self._llm_generate_script(
            action=action,
            gesture=gesture,
            tts_text=text,
            valence=self.last_valence,
            arousal=self.last_arousal,
            tags=self.last_tags
        )

        if not script_json:
            self.get_logger().error("LLM non ha prodotto uno script valido. Fallback TTS.")
            self._tts("Ok.")
            return

        # 2) validate/limit
        ok, reason = self._validate_script(script_json)
        if not ok:
            self.get_logger().error(f"Script rifiutato: {reason}. Fallback TTS.")
            self._tts("Ok.")
            return

        # 3) execute
        try:
            self._execute_script(script_json)
        except Exception as e:
            self.get_logger().error(f"Errore in esecuzione script: {e}")

    # ---------- LLM prompt ----------
    def _llm_generate_script(self, action: str, gesture: str, tts_text: str,
                             valence: float, arousal: float, tags: List[str]) -> Dict[str, Any]:
        tags_str = ", ".join(tags) if tags else "n/a"

        generation_config = {
            "response_mime_type": "application/json",
            "response_schema": {
                "type": "object",
                "properties": {
                    "script_name": {"type": "string"},
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "op": {"type": "string"},
                                "text": {"type": "string"},
                                "name": {"type": "string"},
                                "speed": {"type": "number"},
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "theta": {"type": "number"},
                                "joints": {"type": "array", "items": {"type": "string"}},
                                "angles": {"type": "array", "items": {"type": "number"}},
                                "which": {"type": "string"},
                                "sec": {"type": "number"}
                            },
                            "required": ["op"]
                        }
                    }
                },
                "required": ["steps"]
            }
        }

        prompt = f"""
Sei un planner motorio per il robot NAO. Genera uno SCRIPT JSON (niente testo extra) che realizza l’azione richiesta,
usando SOLO queste primitive consentite:

- "say":        {{ "op":"say", "text":"..." }}
- "posture":    {{ "op":"posture", "name":"StandInit|Stand|SitRelax|Sit", "speed":0.1..1.0 }}
- "move_rel":   {{ "op":"move_rel", "x":-0.5..0.5, "y":-0.3..0.3, "theta":-1..1, "speed":0.1..1.0 }}
- "set_angles": {{ "op":"set_angles", "joints":[...], "angles":[...], "speed":0.1..1.0 }}
- "open_hand":  {{ "op":"open_hand", "which":"R|L" }}
- "close_hand": {{ "op":"close_hand", "which":"R|L" }}
- "sleep":      {{ "op":"sleep", "sec":0.0..3.0 }}

Contesto:
- Azione: "{action}"   (gesture: "{gesture or '-'}")
- Emozioni (1..9): valence={valence:.2f}, arousal={arousal:.2f}, tags=[{tags_str}]

Istruzioni:
- Script breve (<= 20 step), fluido e sicuro.
- Se l'azione è "stop_music": simula avvicinamento + pressione di un tasto e ritorno neutro.
- Se l'azione è "dance": breve sequenza ritmica braccia, non eccessiva.
- Se l'azione è "talk" e c'è testo, inizia con "say".
- Restituisci SOLO JSON con "script_name" (string) e "steps" (array).
"""
        try:
            res = LLM_MODEL.generate_content(prompt, generation_config=generation_config)
            return _loads_json(res.text or "")
        except Exception as e:
            self.get_logger().error(f"LLM error: {e}")
            return {}

    # ---------- Validator / limiter ----------
    def _validate_script(self, script: Dict[str, Any]) -> (bool, str):
        steps = script.get("steps")
        if not isinstance(steps, list):
            return False, "steps mancante o non lista"
        if len(steps) == 0 or len(steps) > MAX_STEPS:
            return False, f"numero steps non valido (1..{MAX_STEPS})"

        total_sleep = 0.0

        for i, st in enumerate(steps):
            if not isinstance(st, dict):
                return False, f"step {i} non è oggetto"
            op = st.get("op")
            if op not in ALLOWED_OPS:
                return False, f"op non consentita: {op}"

            if op == "say":
                if not isinstance(st.get("text",""), str) or not st["text"]:
                    return False, f"step {i} say: text mancante"

            elif op == "posture":
                name = st.get("name","")
                if name not in ("StandInit","Stand","SitRelax","Sit"):
                    return False, f"step {i} posture: name non valido"
                spd = float(st.get("speed", 0.6))
                st["speed"] = max(SPEED_MIN, min(SPEED_MAX, spd))

            elif op == "move_rel":
                x = float(st.get("x", 0.0)); y = float(st.get("y", 0.0)); th = float(st.get("theta",0.0))
                spd = float(st.get("speed", 0.4))
                st["x"] = max(MOVE_X_MIN, min(MOVE_X_MAX, x))
                st["y"] = max(MOVE_Y_MIN, min(MOVE_Y_MAX, y))
                st["theta"] = max(MOVE_TH_MIN, min(MOVE_TH_MAX, th))
                st["speed"] = max(SPEED_MIN, min(SPEED_MAX, spd))

            elif op == "set_angles":
                joints = st.get("joints", [])
                angles = st.get("angles", [])
                spd    = float(st.get("speed", 0.25))
                if not isinstance(joints, list) or not isinstance(angles, list) or len(joints) != len(angles) or len(joints)==0:
                    return False, f"step {i} set_angles: joints/angles invalidi"
                new_angles = []
                for a in angles:
                    try:
                        av = float(a)
                    except Exception:
                        return False, f"step {i} set_angles: angle non numerico"
                    new_angles.append(max(ANGLE_MIN, min(ANGLE_MAX, av)))
                st["angles"] = new_angles
                st["speed"]  = max(SPEED_MIN, min(SPEED_MAX, spd))

            elif op in ("open_hand","close_hand"):
                which = HAND_MAP.get(st.get("which",""), "")
                if which == "":
                    return False, f"step {i} {op}: which non valido (R|L)"
                st["which"] = which

            elif op == "sleep":
                sec = float(st.get("sec", 0.0))
                sec = max(SLEEP_MIN, min(SLEEP_MAX, sec))
                st["sec"] = sec
                total_sleep += sec

        if total_sleep > MAX_TOTAL_SLEEP_SEC:
            return False, "sleep totale eccessivo"

        return True, "ok"

    # ---------- Helpers di esecuzione (qi.Session) ----------
    def _tts(self, text: str):
        if DRY_RUN:
            self.get_logger().info(f"[DRY][TTS] {text}")
            return
        try:
            self.tts.say(text)
        except Exception as e:
            self.get_logger().warn(f"TTS error: {e}")

    def _posture(self, name="StandInit", speed=0.6):
        if DRY_RUN:
            self.get_logger().info(f"[DRY] posture({name}, {speed})")
            return
        try:
            self.posture_service.goToPosture(name, float(speed))
        except Exception as e:
            self.get_logger().warn(f"posture error: {e}")

    def _move_rel(self, x=0.0, y=0.0, theta=0.0):
        if DRY_RUN:
            self.get_logger().info(f"[DRY] move_rel(x={x}, y={y}, th={theta})")
            return
        try:
            # posture per sicurezza prima del movimento
            self.posture_service.goToPosture("StandInit", 0.6)
            self.motion_service.moveTo(float(x), float(y), float(theta))
        except Exception as e:
            self.get_logger().warn(f"moveTo error: {e}")

    def _set_angles(self, joints: List[str], angles: List[float], speed: float):
        if DRY_RUN:
            self.get_logger().info(f"[DRY] set_angles({joints}, {angles}, speed={speed})")
            return
        try:
            self.motion_service.setAngles(joints, [float(a) for a in angles], float(speed))
        except Exception as e:
            self.get_logger().warn(f"setAngles error: {e}")

    def _open_hand(self, which: str):
        if DRY_RUN:
            self.get_logger().info(f"[DRY] open_hand({which})")
            return
        try:
            self.motion_service.openHand(which)
        except Exception as e:
            self.get_logger().warn(f"openHand error: {e}")

    def _close_hand(self, which: str):
        if DRY_RUN:
            self.get_logger().info(f"[DRY] close_hand({which})")
            return
        try:
            self.motion_service.closeHand(which)
        except Exception as e:
            self.get_logger().warn(f"closeHand error: {e}")

    def _sleep(self, sec: float):
        time.sleep(float(sec))

    # ---------- Esecutore della DSL ----------
    def _execute_script(self, script: Dict[str, Any]):
        steps = script.get("steps", [])
        name  = script.get("script_name", "unnamed")
        self.get_logger().info(f"Eseguo script: {name}  ({len(steps)} steps)")

        for i, st in enumerate(steps):
            op = st.get("op")
            if op == "say":
                self._tts(st["text"])

            elif op == "posture":
                posture_name = st["name"]; spd = st.get("speed", 0.6)
                self._posture(posture_name, spd)

            elif op == "move_rel":
                self._move_rel(st["x"], st["y"], st["theta"])

            elif op == "set_angles":
                self._set_angles(st["joints"], st["angles"], st.get("speed", 0.25))

            elif op == "open_hand":
                self._open_hand(st["which"])

            elif op == "close_hand":
                self._close_hand(st["which"])

            elif op == "sleep":
                self._sleep(st["sec"])

            else:
                self.get_logger().warn(f"Step {i}: op sconosciuta {op}, salto.")

# ---- main ----
def main():
    rclpy.init()
    node = ActionCodegenExecutorQi()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
