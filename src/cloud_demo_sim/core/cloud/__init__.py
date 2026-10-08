#
#	Hey MiRo - off-board (P4) cloud clients.
#
#	worker.py  CloudWorker: runs blocking cloud calls off the 50 Hz path
#	llm.py     OpenAI(-compatible) speech-to-text and chat, reply parser,
#	           system prompt and conversation memory
#	tts.py     ElevenLabs text-to-speech (REST, 24 kHz int16)
#	music.py   song manifest, mp3 decoding, optional Spotify lookup
#
#	Everything here is pure Python (numpy, requests, PyYAML, optional
#	openai) and never imports rospy or miro2, so it can be unit tested
#	and reused by the tools/ scripts.
#
