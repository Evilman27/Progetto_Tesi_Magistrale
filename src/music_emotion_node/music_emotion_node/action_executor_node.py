import os, json, time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

DRY_RUN = os.getenv("DRY_RUN", "0") == "1"

# Prova a importare qi (NAOqi SDK nuovo stile)
try:
    import qi
except Exception as e:
    if not DRY_RUN:
        raise
    qi = None  # dry

def _loads(s: str):
    try: return json.loads(s)
    except Exception: return {}

class ActionExecutorQi(Node):
    def __init__(self):
        super().__init__('action_executor_qi')

        # --- parametri ROS2 (default modificabili da CLI) ---
        self.declare_parameter('ip', '192.168.0.101')
        self.declare_parameter('port', 9559)
        ip   = self.get_parameter('ip').get_parameter_value().string_value
        port = self.get_parameter('port').get_parameter_value().integer_value

        # --- connessione qi.Session ---
        self.session = None
        self.motion_service = None
        self.posture_service = None
        self.tts = None
        self.life_service = None
        self.awareness_service = None

        if DRY_RUN:
            self.get_logger().warn("DRY_RUN=1: simulo le azioni senza collegarmi al NAO.")
        else:
            self.session = self._connect_nao(ip, port)
            self.motion_service   = self.session.service("ALMotion")
            self.posture_service  = self.session.service("ALRobotPosture")
            self.tts              = self.session.service("ALTextToSpeech")
            self.life_service     = self.session.service("ALAutonomousLife")
            self.awareness_service= self.session.service("ALBasicAwareness")
            try:
                self.tts.setLanguage("Italian")
            except Exception:
                pass
            try:
                self.motion_service.wakeUp()
            except Exception:
                pass

        # --- subscriber ---
        self.sub = self.create_subscription(String, "/music_emotion/action_json", self.on_action, 10)

        # NEW: TTS diretto
        self.sub_tts = self.create_subscription(String, "/music_emotion/tts", self.on_tts, 10)

        self.get_logger().info(f"[executor] pronto. ip={ip} port={port} dry={DRY_RUN}")

    # ---------- connessione ----------
    def _connect_nao(self, ip, port):
        sess = qi.Session()
        url = f"tcp://{ip}:{port}"
        try:
            sess.connect(url)
            self.get_logger().info(f"Connesso a {url}")
            return sess
        except RuntimeError as e:
            self.get_logger().fatal(f"Connessione fallita a {url}: {e}")
            raise

    # ---------- utils motori ----------
    def _tts(self, text):
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
            self._posture("StandInit", 0.6)
            self.motion_service.moveTo(float(x), float(y), float(theta))
        except Exception as e:
            self.get_logger().warn(f"moveTo error: {e}")

    def _set_angles(self, joints, angles, speed=0.25):
        if DRY_RUN:
            self.get_logger().info(f"[DRY] set_angles({joints}, {angles}, speed={speed})")
            return
        try:
            self.motion_service.setAngles(joints, [float(a) for a in angles], float(speed))
        except Exception as e:
            self.get_logger().warn(f"setAngles error: {e}")

    def _open_hand(self, which="RHand"):
        if DRY_RUN:
            self.get_logger().info(f"[DRY] open_hand({which})")
            return
        try:
            self.motion_service.openHand(which)
        except Exception as e:
            self.get_logger().warn(f"openHand error: {e}")

    def _close_hand(self, which="RHand"):
        if DRY_RUN:
            self.get_logger().info(f"[DRY] close_hand({which})")
            return
        try:
            self.motion_service.closeHand(which)
        except Exception as e:
            self.get_logger().warn(f"closeHand error: {e}")

    def _head_tilt(self, pitch=0.2, speed=0.2):
        self._set_angles(["HeadPitch"], [float(pitch)], float(speed))

    def _wave_arms_once(self, speed=0.25):
        self._set_angles(["LShoulderPitch","RShoulderPitch"], [0.2,0.2], speed);  time.sleep(0.35)
        self._set_angles(["LShoulderPitch","RShoulderPitch"], [1.2,1.2], speed);  time.sleep(0.35)

    def _nod(self, times=2, amp=0.18, speed=0.25):
        for _ in range(times):
            self._set_angles(["HeadPitch"], [amp], speed);    time.sleep(0.25)
            self._set_angles(["HeadPitch"], [-0.05], speed);  time.sleep(0.25)

    # ---------- callback azioni ----------
    def on_action(self, msg: String):
        payload = _loads(msg.data)
        action  = (payload.get("action")  or "").strip().lower()
        gesture = (payload.get("gesture") or "").strip().lower()
        text    = (payload.get("text")    or "").strip()

        if text:
            self._tts(text)  # 👉 pronuncia la frase generata dal live_emotion_pub

        if not action:
            self.get_logger().warn("Azione vuota.")
            return

        self.get_logger().info(f"→ eseguo: {action}  gesture={gesture or '-'}")

        try:
            if action == "stop_music":
                self._do_stop_music()
            elif action == "rest":
                self._do_rest()
            elif action == "talk":
                self._do_talk(text or "Certo. Dimmi pure come ti fa sentire questo brano.")
            elif action == "listen":
                self._do_listen()
            elif action == "play":
                self._do_play()
            elif action == "dance":
                self._do_dance(gesture or "wave_arms")
            elif action == "lower_volume":
                self._do_volume("down")
            elif action == "raise_volume":
                self._do_volume("up")
            elif action == "skip_track":
                self._do_skip_track()
            else:
                self.get_logger().warn(f"Azione sconosciuta: {action} → fallback talk")
                self._do_talk("Ok, resto in ascolto.")
        except Exception as e:
            self.get_logger().error(f"Errore durante esecuzione '{action}': {e}")

    def on_tts(self, msg: String):
        text = (msg.data or "").strip()
        if text:
            self._tts(text)

    # ---------- AZIONI CONCRETE ----------
    def _do_stop_music(self):
        self._tts("Va bene, interrompo la musica.")
        self._posture("StandInit", 0.6)
        self._move_rel(0.25, 0.0, 0.0)
        self._head_tilt(0.1, 0.25)
        # gesto braccio destro “click”
        self._set_angles(["RShoulderPitch","RShoulderRoll","RElbowYaw","RElbowRoll","RWristYaw"], [0.2, 0.0, 0.0, -0.5, 0.0], 0.25)
        self._open_hand("RHand"); time.sleep(0.25)
        self._close_hand("RHand"); time.sleep(0.2)
        self._open_hand("RHand"); time.sleep(0.2)
        self._posture("StandInit", 0.6)

    def _do_rest(self):
        self._tts("Mi fermo un attimo.")
        self._posture("SitRelax", 0.6)

    def _do_talk(self, text):
        self._tts(text if text else "Parliamone pure.")

    def _do_listen(self):
        self._tts("Ti ascolto.")
        self._posture("StandInit", 0.6)
        self._move_rel(0.05, 0.0, 0.0)
        self._head_tilt(0.2, 0.25)
        self._nod(1)

    def _do_play(self):
        self._tts("Giochiamo insieme?")
        try:
            self._set_angles(["LShoulderRoll","RShoulderRoll"], [0.5,-0.5], 0.25); time.sleep(0.35)
            self._set_angles(["LShoulderRoll","RShoulderRoll"], [-0.5,0.5], 0.25); time.sleep(0.35)
        except Exception: pass

    def _do_dance(self, style):
        self._tts("Balliamo!")
        for _ in range(3):
            self._wave_arms_once()

    def _do_volume(self, direction):
        self._tts("Ok.")
        # se riproduci dal NAO, dovresti usare ALAudioDevice; qui tralasciamo

    def _do_skip_track(self):
        self._tts("Cambio brano.")
        # simulazione gesto “click” simile allo stop
        self._posture("StandInit", 0.6)
        self._move_rel(0.15, 0.0, 0.0)
        self._set_angles(["RShoulderPitch","RShoulderRoll","RElbowYaw","RElbowRoll","RWristYaw"], [0.3, 0.0, 0.0, -0.4, 0.0], 0.25)
        self._open_hand("RHand"); time.sleep(0.2)
        self._close_hand("RHand"); time.sleep(0.2)
        self._posture("StandInit", 0.6)

def main():
    rclpy.init()
    node = ActionExecutorQi()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
