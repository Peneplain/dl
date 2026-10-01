#!/usr/bin/env bash
set -euo pipefail

gui_root="${MUJOCO_GUI_ROOT:-/opt/mujoco-gui-root}"
display_num="${MUJOCO_VNC_DISPLAY:-99}"
vnc_port="${MUJOCO_VNC_PORT:-5900}"
display=":${display_num}"
config="$(mktemp /tmp/mujoco-xorg.XXXXXX.conf)"
passwd_file="$(mktemp /tmp/mujoco-vnc-passwd.XXXXXX)"
vnc_password="group3"
printf '%s\n' "$vnc_password" >"$passwd_file"
chmod 600 "$passwd_file"

cat >"$config" <<'EOF'
Section "ServerFlags"
  Option "AutoAddDevices" "False"
  Option "DontVTSwitch" "True"
EndSection
Section "Module"
  Load "glx"
EndSection
Section "Device"
  Identifier "MujocoDummy"
  Driver "dummy"
  VideoRam 256000
EndSection
Section "Monitor"
  Identifier "MujocoMonitor"
  HorizSync 5.0 - 1000.0
  VertRefresh 5.0 - 200.0
  Modeline "1024x768" 65.00 1024 1048 1184 1344 768 771 777 806
EndSection
Section "Screen"
  Identifier "MujocoScreen"
  Device "MujocoDummy"
  Monitor "MujocoMonitor"
  DefaultDepth 24
  SubSection "Display"
    Depth 24
    Modes "1024x768"
  EndSubSection
EndSection
Section "ServerLayout"
  Identifier "MujocoLayout"
  Screen "MujocoScreen"
EndSection
EOF

export DISPLAY="$display"
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export LIBGL_DRIVERS_PATH="${gui_root}/usr/lib/x86_64-linux-gnu/dri"
export __GLX_VENDOR_LIBRARY_NAME=mesa
export PATH="${gui_root}/usr/bin:${PATH}"
export LD_LIBRARY_PATH="${gui_root}/usr/lib/x86_64-linux-gnu:${gui_root}/usr/lib/x86_64-linux-gnu/dri:${gui_root}/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

"${gui_root}/usr/lib/xorg/Xorg" "$display" -config "$config" \
  -modulepath "${gui_root}/usr/lib/xorg/modules" -nolisten tcp -noreset \
  -logfile /tmp/mujoco-xorg.log >/tmp/mujoco-xorg.stdout 2>&1 &
xorg_pid=$!
x11vnc_pid=
cleanup() {
  [[ -z "$x11vnc_pid" ]] || kill "$x11vnc_pid" 2>/dev/null || true
  kill "$xorg_pid" 2>/dev/null || true
  rm -f "$config" "$passwd_file"
}
trap cleanup EXIT

ready=false
for _ in $(seq 1 100); do
  if "${gui_root}/usr/bin/xdpyinfo" -display "$display" >/dev/null 2>&1; then
    ready=true
    break
  fi
  if ! kill -0 "$xorg_pid" 2>/dev/null; then
    cat /tmp/mujoco-xorg.log >&2
    exit 1
  fi
  sleep 0.1
done
if [[ "$ready" != true ]]; then
  cat /tmp/mujoco-xorg.log >&2
  exit 1
fi
"${gui_root}/usr/bin/x11vnc" -display "$display" -rfbport "$vnc_port" \
  -localhost -passwdfile "$passwd_file" -forever -shared -wait 10 -defer 5 -quiet \
  >/tmp/mujoco-x11vnc.log 2>&1 &
x11vnc_pid=$!
vnc_ready=false
for _ in $(seq 1 100); do
  if (echo >/dev/tcp/127.0.0.1/"$vnc_port") >/dev/null 2>&1; then
    vnc_ready=true
    break
  fi
  if ! kill -0 "$x11vnc_pid" 2>/dev/null; then
    cat /tmp/mujoco-x11vnc.log >&2
    exit 1
  fi
  sleep 0.1
done
if [[ "$vnc_ready" != true ]]; then
  cat /tmp/mujoco-x11vnc.log >&2
  exit 1
fi

echo "MuJoCo virtual display ready on ${display}; VNC on 127.0.0.1:${vnc_port}."
echo "VNC password authentication is enabled."
"$@"
