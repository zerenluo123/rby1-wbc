# setting up the cameras

first configure the ip address - set the ethernet port's address to 192.168.88.1

install aravis
https://aravisproject.github.io/aravis/aravis-stable/building.html

```bash
sudo apt update
sudo apt upgrade
sudo apt install -f
sudo apt install libxml2-dev gobject-introspection libgirepository1.0-dev
cd ~/aravis-0.8.35
meson setup buildsudo apt install gstreamer1.0-plugins-bad
cd build
ninja
sudo ninja install
sudo ldconfig
```

test with 
```bash
arv-tool-0.8 list
```

install viewers
```bash
sudo apt install aravis-tools
sudo apt install gstreamer1.0-plugins-bad
```

launch viewer
```bash
arv-viewer-0.8
```

install aravis python bindings
```bash
# Make sure you're in the egoumi environment
conda activate egoumi

# Install PyGObject via conda
conda install -c conda-forge pygobject

# set path
export GI_TYPELIB_PATH=/usr/local/lib/x86_64-linux-gnu/girepository-1.0:$GI_TYPELIB_PATH
```

increase max UDP packet size
ifconfig etho0 mtu 9000 up
or/and
vi /etc/network/interfaces
add mtu 9000

test camera streaming:
```bash
python3 scripts/camera_stream_viewer.py --list
```

Use one `--camera observation_key="DEVICE_ID"` flag per camera, where `DEVICE_ID`
is exactly the identifier returned by the previous command (e.g.
`FLIR-Blackfly S BFS-PGE-50S5C-25260985`):

```bash
python3 scripts/camera_stream_viewer.py \
  --camera camera_head_main_rgb="FLIR-Blackfly S BFS-PGE-50S5C-25260985" \
  --camera camera_left_main_rgb="FLIR-Blackfly S BFS-PGE-23S3C-24260091"
```

When hardware is unavailable you can still validate the pipeline by generating
synthetic images:

```bash
python3 scripts/camera_stream_viewer.py --mock
```

```
sudo /usr/local/sbin/setup-x710.sh
```