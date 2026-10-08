# MiRo MDK tarball (optional)

`docker/Dockerfile` installs the MiRo Developer Kit (MDK) 2-230105. By default it
downloads `mdk_2-230105.tgz` from the Google Drive copy used by the University of
Sheffield COM3528 install script. To build without that download, put the tarball
here as `docker/mdk/mdk_2-230105.tgz` (it is gitignored and is bind-mounted during
the build, so it never becomes an image layer), or pass `--build-arg MDK_URL=...`.

The MDK is licensed by Consequential Robotics; do not commit it to this repository.
