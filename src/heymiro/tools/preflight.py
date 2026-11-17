#!/usr/bin/env python3
#
#	Pre-flight check run by run.sh before the demo starts.
#
#	Prints which platform overlay and AI pipeline will be used, the features
#	that are switched off (and why), and - for the offline pipeline - whether
#	the docker/local-ai services answer. Never prints a secret value.
#
#	Exit status: 0 = ready, 1 = the offline services are not reachable.
#
#	usage: tools/preflight.py [--timeout S]
#

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))

import heymiro_config  # noqa: E402


def service_ok(base_url, timeout_s):

	try:
		import requests
	except ImportError:
		return None
	base = base_url[:-3] if base_url.endswith("/v1") else base_url
	try:
		return requests.get(base.rstrip("/") + "/health", timeout=timeout_s).status_code == 200
	except Exception:
		return False


def main():

	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--timeout", type=float, default=3.0)
	args = ap.parse_args()

	hm = heymiro_config.load()
	cfg = hm.cfg
	offline = hm.local
	print("Hey MiRo pre-flight")
	print("  platform:    %s  (config/platform/%s.yaml; speech from %s)" % (
		cfg.get("platform"), hm.platform_name(), cfg.get("hearing.input_topic")))
	print("  AI pipeline: %s" % ("offline (local models, docker/local-ai)" if offline
		else "online (OpenAI, ElevenLabs, Picovoice)"))
	f = hm.features
	print("  features:    conversation=%s wake_word=%s tts=%s vad=%s" % (
		f.get("conversation"), f.get("wake_word"), f.get("tts"), f.get("vad")))
	for w in hm.warnings:
		print("  note: " + w)

	if offline:
		ready = True
		for name, route in (("llm", hm.route("llm")), ("speech", hm.route("stt"))):
			ok = service_ok(route.base_url, args.timeout)
			if ok is None:
				print("  (python3-requests missing: cannot check the local services)")
				break
			print("  local %-6s %s" % (name, "ready" if ok else "NOT reachable at " + route.base_url))
			ready = ready and ok
		if not ready:
			print("")
			print("The offline pipeline needs the local AI services. On the host run:")
			print("  docker/local-ai/download_models.bash   # once (about 5 GB)")
			print("  docker/local-ai/up.bash                # every boot")
			print("or start the container with: docker/run_docker.bash ... --offline")
			print("(skip this check with run.sh --no-check)")
			return 1
	return 0


if __name__ == "__main__":
	sys.exit(main())
