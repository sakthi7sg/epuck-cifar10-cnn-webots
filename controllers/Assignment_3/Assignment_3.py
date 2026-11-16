"""E-puck search with colour-block detection and CNN object classification (Webots, Python).

One arena holds a red, a blue and a green block (0.3 x 0.3 x 0.1 m, placed at random)
and animals (a dog, a deer and a horse). The e-puck searches the arena and:

  * colour blocks: prints "I see <colour>" on the FIRST encounter of each colour,
    then a summary of the colours found so far (as in Part 1);
  * animals: when it faces an animal, it classifies the camera image with the
    CIFAR-10 CNN trained in notebooks/Assignment_3_CNN.ipynb and prints the prediction;
  * every detection is saved as a camera image in the captures folder.

Optimisations so the CNN does not disturb the robot's movement:
  * the CNN runs in a background thread, so the 64 ms control loop never waits;
  * the CNN only runs when the robot is facing an animal (supervisor photo trigger),
    not on every frame;
  * PyTorch uses one CPU thread; optional static int8 quantisation (USE_INT8).

Setup: the e-puck must have supervisor TRUE. For the CNN, put cifar10_cnn.pth in this
folder and install PyTorch in the Python that Webots uses (pip install torch). Without
them the colour search and the image captures still work.
"""
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

from controller import Camera, Supervisor

# ---------------------------------------------------------------- settings
TIME_STEP = 64                  # control step (ms)
SPEED = 5.0                     # base wheel speed (rad/s); e-puck maximum is 6.28
SCAN_STEPS = 28                 # steps for one full turn on the spot at SPEED
MAX_SPEED = 6.28
OBSTACLE = 80                   # proximity value meaning "something is close"
ARENA_HALF = 1.2                # blocks are placed within +/- this distance of the centre (m)
SEED = None                     # an integer gives a repeatable block layout
TIME_LIMIT = 480.0              # stop after this many simulated seconds (s)

COLOURS = {"red": (1, 0, 0), "blue": (0, 0, 1), "green": (0, 1, 0)}
ANIMALS = ("Dog", "Cat", "Horse", "Deer")       # Webots PROTO names (also CIFAR-10 classes)
PHOTO_RANGE = (0.35, 0.7)       # take an animal photo between these distances (m)
PHOTO_BEARING = 0.12            # ... when the animal is this close to straight ahead (rad)
MIN_CONFIDENCE = 0.40           # below this the CNN result is reported as "not sure"
USE_INT8 = False                # True: static int8 quantisation (faster, about 4x smaller)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "cifar10_cnn.pth")
CAPTURE_DIR = os.path.join(HERE, "captures")


# ---------------------------------------------------------------- 1. arena set-up
def find_animals(robot):
    """Return {name: (x, y)} of the animal models in the world."""
    animals = {}
    children = robot.getRoot().getField("children")
    for i in range(children.getCount()):
        node = children.getMFNode(i)
        if node.getTypeName() in ANIMALS:
            x, y = node.getField("translation").getSFVec3f()[:2]
            animals[node.getTypeName().lower()] = (x, y)
    return animals


def place_blocks(robot, keep_clear):
    """Move (or create) the three blocks to random free positions.

    A position is free if it is at least 0.5 m from the robot, 0.55 m from the
    other blocks and 0.45 m from every animal.
    """
    start = robot.getSelf().getField("translation").getSFVec3f()[:2]
    children = robot.getRoot().getField("children")
    placed = []
    for colour, (r, g, b) in COLOURS.items():
        name = colour.upper() + "_BLOCK"
        for _ in range(2000):
            p = (random.uniform(-ARENA_HALF, ARENA_HALF), random.uniform(-ARENA_HALF, ARENA_HALF))
            if (math.dist(p, start) < 0.5 or any(math.dist(p, q) < 0.55 for q in placed)
                    or any(math.dist(p, a) < 0.45 for a in keep_clear)):
                continue
            node = robot.getFromDef(name)
            if node is None:
                children.importMFNodeFromString(-1, f"""
                    DEF {name} Solid {{
                      translation {p[0]} {p[1]} 0.05
                      name "{name}"
                      children [ Shape {{
                        appearance PBRAppearance {{ baseColor {r} {g} {b} roughness 1 metalness 0 }}
                        geometry Box {{ size 0.3 0.3 0.1 }} }} ]
                      boundingObject Box {{ size 0.3 0.3 0.1 }}
                    }}""")
                node = robot.getFromDef(name)
            if node is None:
                print(f"WARNING: could not create the {colour} block")
                break
            node.getField("translation").setSFVec3f([p[0], p[1], 0.05])
            placed.append(p)
            print(f"{colour.capitalize():5s} block at x = {p[0]:5.2f} m, y = {p[1]:5.2f} m")
            break
        else:
            print(f"WARNING: no free position found for the {colour} block")


# ---------------------------------------------------------------- 2. colour detection
def classify_pixel(r, g, b):
    """Return the block colour of one pixel, or None. The main channel must be
    bright (> 80) and the other two below 30 % of it, so shading does not matter."""
    if r > 80 and g < 0.3 * r and b < 0.3 * r:
        return "red"
    if g > 80 and r < 0.3 * g and b < 0.3 * g:
        return "green"
    if b > 80 and r < 0.3 * b and g < 0.3 * b:
        return "blue"
    return None


def scan_colours(image, w, h):
    """Count block-coloured pixels: whole image, their x sum, and the 17 x 13 centre window."""
    count = {c: 0 for c in COLOURS}
    sum_x = {c: 0 for c in COLOURS}
    centre = {c: 0 for c in COLOURS}
    for x in range(w):
        for y in range(h):
            c = classify_pixel(Camera.imageGetRed(image, w, x, y),
                               Camera.imageGetGreen(image, w, x, y),
                               Camera.imageGetBlue(image, w, x, y))
            if c:
                count[c] += 1
                sum_x[c] += x
                if abs(x - w // 2) <= 8 and abs(y - h // 2) <= 6:
                    centre[c] += 1
    return count, sum_x, centre


# ---------------------------------------------------------------- 3. CNN
def load_cnn():
    """Load the trained CNN. Returns (model, torch) or (None, None) with a message."""
    try:
        import torch
        from cnn_model import CifarCNN
    except ImportError as err:
        print(f"[CNN] PyTorch is not available in Webots' Python ({err}).")
        print("[CNN] Animal photos are still saved; install PyTorch (pip install torch) for predictions.")
        return None, None
    if not os.path.isfile(MODEL_PATH):
        print(f"[CNN] Model file not found: {MODEL_PATH}")
        print("[CNN] Animal photos are still saved; add cifar10_cnn.pth for predictions.")
        return None, None
    torch.set_num_threads(1)
    model = CifarCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    if USE_INT8:
        from torch.ao.quantization import get_default_qconfig_mapping
        from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
        engines = torch.backends.quantized.supported_engines
        torch.backends.quantized.engine = "x86" if "x86" in engines else engines[-1]
        example = torch.randn(1, 3, 32, 32)
        model = prepare_fx(model, get_default_qconfig_mapping(torch.backends.quantized.engine), (example,))
        with torch.inference_mode():
            for _ in range(20):
                model(torch.randn(16, 3, 32, 32))      # calibration (real images give better accuracy)
        model = convert_fx(model)
    with torch.inference_mode():                       # warm-up, so the first real call is fast
        for _ in range(3):
            model(torch.randn(1, 3, 32, 32))
    print(f"[CNN] Model loaded ({'int8' if USE_INT8 else 'float32'}).")
    return model, torch


def camera_to_rgb(camera, torch):
    """Webots camera image (BGRA bytes) -> H x W x 3 uint8 RGB tensor."""
    w, h = camera.getWidth(), camera.getHeight()
    bgra = torch.frombuffer(bytearray(camera.getImage()), dtype=torch.uint8).view(h, w, 4)
    return bgra[:, :, [2, 1, 0]].contiguous()


def cnn_predict(model, torch, rgb):
    """Classify one frame. Returns (class, probability, CPU time in ms)."""
    from cnn_model import CLASSES, preprocess
    start = time.thread_time()          # CPU time of this thread, not time spent waiting
    with torch.inference_mode():
        probs = torch.softmax(model(preprocess(rgb)), dim=1)[0]
    conf, idx = probs.max(dim=0)
    return CLASSES[int(idx)], float(conf), (time.thread_time() - start) * 1000


# ---------------------------------------------------------------- 4. reports and captures
class Log:
    """Keeps what has been found and saves a camera image for every detection."""

    def __init__(self, camera):
        self.camera = camera
        self.colours = []           # block colours in the order first seen
        self.animals = {}           # animal -> CNN prediction text
        self.block_shots = {}       # colour -> image saved while approaching the block
        self.saved = 0
        os.makedirs(CAPTURE_DIR, exist_ok=True)

    def capture(self, label, t):
        self.saved += 1
        name = f"{self.saved:02d}_{label}_{t:.1f}s.png"
        self.camera.saveImage(os.path.join(CAPTURE_DIR, name), 100)
        return name

    def approach_shot(self, colour, t):
        """Save one image of a new block while it is still a few centimetres away,
        so the picture shows the block in the arena and not only its colour."""
        if colour not in self.block_shots:
            self.block_shots[colour] = self.capture(colour, t)

    def report_colour(self, colour, t):
        previous = list(self.colours)
        self.colours.append(colour)
        missing = [c for c in COLOURS if c not in self.colours]
        name = self.block_shots.get(colour) or self.capture(colour, t)
        print(f"\nI see {colour}")
        print("Previously encountered:", ", ".join(previous) or "None")
        print("Encountered so far:", ", ".join(self.colours))
        print("Not yet found:", ", ".join(missing) or "None")
        print(f"Simulation time: {t:.1f} s   (image: captures/{name})")

    def report_animal(self, animal, t, result=None):
        name = self.capture(animal, t)
        if result is None:
            text = "no CNN prediction (model not loaded)"
        else:
            label, conf, ms = result
            verdict = "correct" if label == animal else "wrong"
            sure = "" if conf >= MIN_CONFIDENCE else ", not sure"
            text = f"CNN says {label} ({conf * 100:.0f}%{sure}, {verdict}, {ms:.1f} ms)"
        self.animals[animal] = text
        print(f"\n[{t:6.1f} s] Photo of the {animal}: {text}   (image: captures/{name})")

    def summary(self):
        print("\n================ SUMMARY ================")
        print("Colour blocks found:", ", ".join(self.colours) or "None")
        for animal in sorted(self.animals):
            print(f"  {animal:5s}: {self.animals[animal]}")
        print(f"Images saved in: {CAPTURE_DIR}")


# ---------------------------------------------------------------- 5. main loop (state machine)
def heading_of(robot):
    """Robot heading in the world (rad): the direction its camera points."""
    m = robot.getSelf().getOrientation()
    return math.atan2(m[3], m[0])


def main():
    robot = Supervisor()
    random.seed(SEED)
    animals = find_animals(robot)
    place_blocks(robot, keep_clear=list(animals.values()))
    print("Animals in the arena:", ", ".join(animals) or "none")

    camera = robot.getDevice("camera")
    camera.enable(TIME_STEP)
    w, h = camera.getWidth(), camera.getHeight()
    sensors = [robot.getDevice(f"ps{i}") for i in range(8)]
    for s in sensors:
        s.enable(TIME_STEP)
    left, right = robot.getDevice("left wheel motor"), robot.getDevice("right wheel motor")
    for motor in (left, right):
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)

    model, torch = load_cnn()
    worker = ThreadPoolExecutor(max_workers=1)      # the CNN runs here, never in the loop
    pending = None                                  # (animal, future) of a running prediction
    log = Log(camera)

    state, counter, direction = "SEARCH", 0, 1      # SEARCH, BACK_OFF, TURN, DONE
    last_scan = 0.0                                 # time of the last look-around turn
    detour = 0                                      # steps to drive straight after avoiding something
    last_pos, last_check = None, 0.0                # stuck watchdog
    report_window = 0.5 * 17 * 13                   # half the centre window must be the colour

    while robot.step(TIME_STEP) != -1:
        t = robot.getTime()
        if state == "DONE":
            continue

        # Collect a finished CNN prediction without waiting for it.
        if pending is not None and pending[1].done():
            log.report_animal(pending[0], pending[2], pending[1].result())
            pending = None

        all_colours = len(log.colours) == len(COLOURS)
        all_animals = set(log.animals) >= set(animals) and pending is None
        if (all_colours and all_animals) or t > TIME_LIMIT:
            left.setVelocity(0.0); right.setVelocity(0.0)
            if t > TIME_LIMIT:
                print(f"\nTime limit of {TIME_LIMIT:.0f} s reached.")
            log.summary()
            state = "DONE"
            continue

        ps = [s.getValue() for s in sensors]
        front = any(ps[i] > OBSTACLE for i in (0, 1, 6, 7))
        image = camera.getImage()
        if image is None:
            continue
        count, sum_x, centre = scan_colours(image, w, h)
        unfound = [c for c in COLOURS if c not in log.colours]
        for c in unfound:                       # picture of a new block on the way to it
            others = sum(count[k] for k in COLOURS if k != c)
            if (0.12 * w * h <= count[c] <= 0.40 * w * h           # whole block in view
                    and abs(sum_x[c] / count[c] - (w - 1) / 2) <= 0.2 * w   # near the centre
                    and others <= 0.03 * w * h):                     # no other block in front
                log.approach_shot(c, t)

        # Animal photo trigger: facing an animal that has no photo yet, at a good distance.
        x, y = robot.getSelf().getPosition()[:2]
        heading = heading_of(robot)
        targets = []
        for animal, (ax, ay) in animals.items():
            if animal in log.animals or (pending and pending[0] == animal):
                continue
            dist = math.hypot(ax - x, ay - y)
            bearing = math.atan2(math.sin(math.atan2(ay - y, ax - x) - heading),
                                 math.cos(math.atan2(ay - y, ax - x) - heading))
            targets.append((dist, bearing, animal))
            if PHOTO_RANGE[0] <= dist <= PHOTO_RANGE[1] and abs(bearing) <= PHOTO_BEARING:
                if model is None:
                    log.report_animal(animal, t)
                elif pending is None:
                    pending = (animal, worker.submit(cnn_predict, model, torch, camera_to_rgb(camera, torch)), t)

        # Stuck watchdog: moved less than 15 cm in 8 s -> back off, turn round and drive away.
        if t - last_check >= 8.0:
            if (state == "SEARCH" and last_pos
                    and math.hypot(x - last_pos[0], y - last_pos[1]) < 0.08):
                state, counter, direction = "BACK_OFF", 20, random.choice([-1, 1])
                detour, last_scan = 45, t
            last_pos, last_check = (x, y), t

        # Block report: close to a block of a colour not seen before.
        if state == "SEARCH" and front and unfound and t > 1.0:   # sensors settle first
            c = max(unfound, key=lambda k: centre[k])
            if centre[c] >= report_window:
                log.report_colour(c, t)
                state, counter, direction = "BACK_OFF", 12, random.choice([-1, 1])

        # Movement
        seen = [c for c in unfound if count[c] >= 8]            # new block colours in view
        if state == "BACK_OFF":
            l = r = -SPEED
            counter -= 1
            if counter == 0:
                state, counter = "TURN", random.randint(10, 25)
        elif state == "TURN":
            l, r = direction * SPEED, -direction * SPEED
            counter -= 1
            if seen:                                            # a new colour came into view
                state, detour = "SEARCH", 0
            elif counter == 0:
                state, detour = "SEARCH", 20                    # then drive straight for a moment
        elif any(ps[i] > OBSTACLE for i in (5, 6, 7)):          # obstacle on the left
            l, r = SPEED, -SPEED
            detour = 20
        elif any(ps[i] > OBSTACLE for i in (0, 1, 2)):          # obstacle on the right
            l, r = -SPEED, SPEED
            detour = 20
        elif seen:                                              # steer to a new block colour
            c = max(seen, key=lambda k: count[k])
            error = (sum_x[c] / count[c] - (w - 1) / 2) / (w / 2)
            l, r = SPEED * (1 + 0.8 * error), SPEED * (1 - 0.8 * error)
        elif detour:                                            # drive past the obstacle first
            l = r = SPEED
            detour -= 1
        else:
            near = [tg for tg in targets if tg[0] > PHOTO_RANGE[0]]
            if near:                                            # steer to the nearest animal without a photo
                dist, bearing, _ = min(near)
                turn = max(-1.0, min(1.0, 2.0 * bearing))
                base = SPEED if abs(bearing) < 0.5 else 0.0     # turn on the spot first
                l, r = base - turn * SPEED, base + turn * SPEED
            elif t - last_scan > 6.0:                           # look all around every 6 s
                state, counter, direction = "TURN", SCAN_STEPS, random.choice([-1, 1])
                last_scan = t
                l, r = direction * SPEED, -direction * SPEED
            else:
                l = r = SPEED

        left.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, l)))
        right.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, r)))

    worker.shutdown(wait=False)


if __name__ == "__main__":
    main()
