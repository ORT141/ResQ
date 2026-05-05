#!/bin/bash
# =============================================================================
# start_nav.sh — Головний скрипт запуску навігації робота
# =============================================================================
# Робот: 15x20 см, LiDAR LD06, Raspberry Pi 5, ROS 2 Jazzy
#
# Розподіл по ядрах (4 ядра: CPU0–CPU3):
#   CPU 0 — Камера + Coral TPU детекція (єдиний вузол)
#   CPU 1 — LiDAR LD06 + Лазерний далекомір + Одометрія + TF
#   CPU 2 — Мережа: Foxglove Bridge + ZMQ клієнт
#   CPU 3 — Nav2 Bringup (SLAM + планувальник + контролер)
# =============================================================================

source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

# =============================================================================
# МЕНЮ ЗАПУСКУ
# =============================================================================
echo ""
echo "═════════════════════════════════════════"
echo " 🤖  Налаштування запуску"
echo "═════════════════════════════════════════"
read -r -p " [Глибина] Використовувати MiDaS сервер (ZMQ/карта глибини)? [y/N]: " USE_MIDAS
echo "═════════════════════════════════════════"
echo ""

# =============================================================================
# ЗАЩИТА ОТ ДВОЙНОГО ЗАПУСКА + ОЧИСТКА ZOMBIE-ПРОЦЕССОВ
# =============================================================================
echo "========================================="
echo " Очищення старих процесів..."
echo "========================================="

# Зупиняємо всі старі екземпляри наших процесів
pkill -f "camera_coral_node"     2>/dev/null || true
pkill -f "audio_fixation_node"   2>/dev/null || true
pkill -f "vital_radar.py"          2>/dev/null || true
pkill -f "ldlidar_stl_ros2"      2>/dev/null || true
pkill -f "foxglove_bridge"       2>/dev/null || true
pkill -f "fake_odom.py"          2>/dev/null || true
pkill -f "nav2_bringup"          2>/dev/null || true
pkill -f "bringup_launch.py"     2>/dev/null || true
pkill -f "slam_toolbox"          2>/dev/null || true
pkill -f "rpi_client.py"         2>/dev/null || true
pkill -f "radar_node.py"         2>/dev/null || true

# Чекаємо поки порти та пристрої звільняться
echo "Очікування звільнення портів (3 сек)..."
sleep 3

# Перевіряємо що порт 8765 (Foxglove) вільний
if ss -tlnp | grep -q ':8765'; then
    echo "[WARN] Порт 8765 ще зайнятий! Примусове очищення..."
    fuser -k 8765/tcp 2>/dev/null || true
    sleep 1
fi

# =============================================================================
# Масив PID для коректного завершення
# =============================================================================
PIDS=()

cleanup() {
    echo ""
    echo "========================================="
    echo " Завершення всіх процесів..."
    echo "========================================="
    # Спочатку м'яко
    for pid in "${PIDS[@]}"; do
        kill -SIGTERM "$pid" 2>/dev/null || true
    done
    sleep 2
    # Потім примусово
    for pid in "${PIDS[@]}"; do
        kill -SIGKILL "$pid" 2>/dev/null || true
    done
    # Фінальне очищення
    pkill -f "camera_coral_node"      2>/dev/null || true
    pkill -f "vital_radar.py"          2>/dev/null || true
    pkill -f "foxglove_bridge"        2>/dev/null || true
    pkill -f "nav2_bringup"           2>/dev/null || true
    pkill -f "bringup_launch.py"      2>/dev/null || true
    pkill -f "slam_toolbox"           2>/dev/null || true
    pkill -f "rpi_client.py"          2>/dev/null || true
    pkill -f "radar_node.py"          2>/dev/null || true
    echo "Всі процеси завершено."
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

echo "========================================="
echo " Nav2 + SLAM  (робот 15x20 см, LD06)"
echo " Прив'язка блоків до ядер CPU 0–3"
echo "========================================="

# =============================================================================
# CPU 0 — Камера + Coral TPU детекція (SSD MobileNet + MoveNet скелет)
# =============================================================================
echo "[CPU 0] Запуск Камера + Coral TPU..."
taskset -c 0 ros2 run usb_camera_driver camera_coral_node &
PIDS+=($!)
echo "        PID=$! → /dev/video0 + EdgeTPU (CPU 0)"

# =============================================================================
# CPU 1 — Сенсори: LiDAR, Лазерний далекомір, Одометрія, TF
# =============================================================================
echo "[CPU 1] Запуск LiDAR LD06..."
taskset -c 1 ros2 launch ldlidar_stl_ros2 ld06.launch.py &
PIDS+=($!)
echo "        PID=$! → /dev/ttyACM* (CPU 1)"

echo "[CPU 1] Запуск радара HLK-LD6002 (дихання/пульс) на ttyAMA0..."
taskset -c 1 python3 /home/r1/ros2_ws/vital_radar.py \
    --ros-args -p port:=/dev/ttyAMA0 &
PIDS+=($!)
echo "        PID=$! → /dev/ttyAMA0 (CPU 1) → /radar/vitals"

echo "[CPU 1] Запуск радара HLK-LD2450..."
taskset -c 1 python3 /home/r1/ros2_ws/radar_node.py \
    --ros-args -p port:=/dev/ttyUSB0 &
PIDS+=($!)
echo "        PID=$! → /dev/ttyUSB0 (CPU 1)"

echo "[CPU 1] Запуск фіктивної одометрії (fake_odom.py)..."
taskset -c 1 python3 /home/r1/ros2_ws/fake_odom.py &
PIDS+=($!)
echo "        PID=$! → /odom + TF odom→base_footprint (CPU 1)"

echo "[CPU 1] Налаштування TF ланцюжка (base_footprint→base_link)..."
taskset -c 1 ros2 run tf2_ros static_transform_publisher \
    --x 0 --y 0 --z 0 --yaw 0 --pitch 0 --roll 0 \
    --frame-id base_footprint --child-frame-id base_link &
PIDS+=($!)

taskset -c 1 ros2 run tf2_ros static_transform_publisher \
    --x 0.1 --y 0 --z 0.1 --yaw 0 --pitch 0 --roll 0 \
    --frame-id base_link --child-frame-id radar_link &
PIDS+=($!)

# =============================================================================
# CPU 2 — Мережа: Foxglove Bridge + ZMQ клієнт
# =============================================================================
echo "[CPU 2] Запуск Foxglove Bridge (порт 8765)..."
taskset -c 2 ros2 launch foxglove_bridge foxglove_bridge_launch.xml \
    send_buffer_limit:=1000000 &
PIDS+=($!)
echo "        PID=$! → ws://0.0.0.0:8765 (CPU 2)"

echo "[CPU 2] Запуск ZMQ-клієнта (rpi_client.py)..."
if [[ "$USE_MIDAS" =~ ^[Yy]$ ]]; then
    taskset -c 2 python3 /home/r1/rpi_client.py &
    PIDS+=($!)
    echo "        PID=$! → ZMQ TX:5555 RX:5556 (CPU 2)"
else
    echo "        [SKIP] MiDaS вимкнено → +ресурси CPU"
fi

# --- VAD (опціонально) ---
if /home/r1/ros2_ws/venv/bin/python3 -c "import alsaaudio" 2>/dev/null; then
    echo "[CPU 1] Запуск детекції голосу (Silero VAD)..."
    taskset -c 1 ros2 run usb_camera_driver audio_fixation_node &
    PIDS+=($!)
    echo "        PID=$! → аудіо (CPU 1)"
else
    echo "[CPU 1] [SKIP] Silero VAD — немає модуля alsaaudio у venv"
fi

# =============================================================================
# Ожидание инициализации сенсоров
# =============================================================================
echo ""
echo "Очікування ініціалізації сенсорів (4 сек)..."
sleep 4

# =============================================================================
# CPU 3 — Nav2 Bringup (SLAM + планувальник + контролер)
# =============================================================================
echo "[CPU 3] Запуск Nav2 (режим SLAM)..."
echo "        SLAM=True, params=/home/r1/ros2_ws/nav2_params.yaml"
echo ""
taskset -c 3 ros2 launch nav2_bringup bringup_launch.py \
    use_sim_time:=false \
    autostart:=true \
    slam:=True \
    params_file:=/home/r1/ros2_ws/nav2_params.yaml &
NAV2_PID=$!
PIDS+=($NAV2_PID)

echo "========================================="
echo " Всі блоки запущено!"
echo " CPU 0: Камера + Coral TPU (SSD + MoveNet)"
echo " CPU 1: LiDAR + Далекомір + Одометрія + TF"
echo " CPU 2: Foxglove + ZMQ клієнт"
echo " CPU 3: Nav2 SLAM"
echo " Foxglove:    ws://0.0.0.0:8765"
echo " ZMQ клієнт: TX→PC:5555  RX←PC:5556"
echo " ROS топіки: /camera/depth/compressed"
echo "             /camera/points"
echo "             /radar/visualization"
echo " Ctrl+C для завершення"
echo "========================================="

# Чекаємо завершення Nav2
wait $NAV2_PID
