#!/usr/bin/env python3
#
#	The two projects (src/cloud_demo_robot and src/cloud_demo_sim) are
#	deliberately self-contained copies of the same code: only their
#	settings, launcher and README differ. This check fails if any other
#	file differs, so a fix made in one copy is not forgotten in the other.
#
#	usage: tools/check_copies.py [--fix-from robot|sim]
#
#	--fix-from robot   copy every shared file from the robot copy to the sim copy
#

import argparse
import filecmp
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBOT = os.path.join(ROOT, "src", "cloud_demo_robot")
SIM = os.path.join(ROOT, "src", "cloud_demo_sim")

# files allowed to differ (or to exist in only one copy)
ALLOWED = {
	"README.md",
	"config/heymiro.yaml",
	"run_robot.sh",
	"run_sim.sh",
}

IGNORED_DIRS = {"__pycache__", ".pytest_cache", "logs"}


def walk(base):

	files = set()
	for dirpath, dirnames, filenames in os.walk(base):
		dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
		for name in filenames:
			if name.endswith(".pyc") or name == "heymiro.local.yaml":
				continue
			files.add(os.path.relpath(os.path.join(dirpath, name), base))
	return files


def main():

	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--fix-from", choices=["robot", "sim"])
	args = parser.parse_args()

	robot = walk(ROBOT)
	sim = walk(SIM)
	problems = []

	for rel in sorted(robot | sim):
		if rel in ALLOWED:
			continue
		a = os.path.join(ROBOT, rel)
		b = os.path.join(SIM, rel)
		if rel not in sim:
			problems.append(("only in robot", rel))
		elif rel not in robot:
			problems.append(("only in sim", rel))
		elif not filecmp.cmp(a, b, shallow=False):
			problems.append(("differs", rel))

	if args.fix_from and problems:
		src, dst = (ROBOT, SIM) if args.fix_from == "robot" else (SIM, ROBOT)
		for kind, rel in problems:
			s = os.path.join(src, rel)
			d = os.path.join(dst, rel)
			if os.path.isfile(s):
				os.makedirs(os.path.dirname(d), exist_ok=True)
				shutil.copy2(s, d)
				print("copied", rel)
			elif os.path.isfile(d):
				os.remove(d)
				print("removed", rel)
		return main_check_only()

	for kind, rel in problems:
		print("%-14s %s" % (kind, rel))
	if problems:
		print("\n%d shared file(s) out of sync; apply the change to both copies "
			"(or run: tools/check_copies.py --fix-from robot)" % len(problems))
		return 1
	print("robot and sim copies are in sync (%d shared files)" % len((robot & sim) - ALLOWED))
	return 0


def main_check_only():

	sys.argv = sys.argv[:1]
	return main()


if __name__ == "__main__":
	sys.exit(main())
