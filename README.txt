Come fare partire il progetto:

pip install -r requirements.txt
pip install music2emo (oppure seguire le indicazioni sul GitHub di music2emo)


su un terminale eseguire:

colcon build
source install/setup.bash
ros2 run music_emotion_node ros2_bridge

su un altro terminale:

colcon build
source install/setup.bash
ros2 run music_emotion_node ActionExecutorQi oppure ActionCodegenExecutorQi

Caldamente consigliato l'uso di un microfono esterno.
Il  software sarà in ascolto per circa 50 secondi, al termine dei quali manderà un errore perché manca la API di Google da aggiungere per usare l'LLM.