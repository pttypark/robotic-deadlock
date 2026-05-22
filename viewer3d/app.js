import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const TILE = 1;
const ROBOT_HEIGHT = 0.32;
const TRACE_URL = new URLSearchParams(window.location.search).get("trace") || "./data/fcfs_trace.json";

const COLORS = {
  ground: 0xe8edf3,
  blocked: 0xd3dae3,
  road: 0xf9fbfd,
  start: 0x86d59a,
  exit: 0x527fc2,
  waiting: 0xf0c45c,
  conflict: 0xe58e70,
  laneLine: 0xaab4c2,
  textDark: 0x253140,
  sharedRing: 0x111827,
};

const els = {
  canvas: document.querySelector("#scene"),
  runLabel: document.querySelector("#runLabel"),
  playButton: document.querySelector("#playButton"),
  prevButton: document.querySelector("#prevButton"),
  nextButton: document.querySelector("#nextButton"),
  resetButton: document.querySelector("#resetButton"),
  topButton: document.querySelector("#topButton"),
  orbitButton: document.querySelector("#orbitButton"),
  stepSlider: document.querySelector("#stepSlider"),
  stepOutput: document.querySelector("#stepOutput"),
  speedSlider: document.querySelector("#speedSlider"),
  speedOutput: document.querySelector("#speedOutput"),
  robotSelect: document.querySelector("#robotSelect"),
  policyStat: document.querySelector("#policyStat"),
  sharedStat: document.querySelector("#sharedStat"),
  queueStat: document.querySelector("#queueStat"),
  doneStat: document.querySelector("#doneStat"),
  legend: document.querySelector("#legend"),
  message: document.querySelector("#message"),
};

let trace;
let scene;
let camera;
let renderer;
let controls;
let pathLine;
let sharedPulse = [];
let frameIndex = 0;
let playing = false;
let selectedRobotId = "";
let lastTick = 0;
let robotMeshes = new Map();
let nodeById = new Map();
let allRobotIds = [];

init().catch((error) => {
  console.error(error);
  els.message.textContent = `3D viewer failed to start: ${error.message}`;
});

async function init() {
  trace = await loadTrace();
  nodeById = new Map(trace.layout.nodes.map((node) => [node.id, node]));
  allRobotIds = collectRobotIds(trace.frames);
  selectedRobotId = allRobotIds[0] || "";

  setupRenderer();
  setupScene();
  buildLayout();
  buildRobots();
  buildLegend();
  bindControls();
  configureUi();
  goToFrame(0);
  animate(0);
}

async function loadTrace() {
  const response = await fetch(TRACE_URL, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Could not load ${TRACE_URL}`);
  }
  return response.json();
}

function setupRenderer() {
  renderer = new THREE.WebGLRenderer({
    canvas: els.canvas,
    antialias: true,
    alpha: false,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  window.addEventListener("resize", resizeRenderer);
}

function setupScene() {
  const { rows, cols } = trace.layout.grid_size;
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xeef2f6);

  camera = new THREE.PerspectiveCamera(48, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.set(cols * 0.7, Math.max(rows, cols) * 0.9, rows * 0.8);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 0, 0);
  controls.minDistance = 5;
  controls.maxDistance = Math.max(rows, cols) * 2.2;
  controls.maxPolarAngle = Math.PI * 0.48;

  const hemi = new THREE.HemisphereLight(0xffffff, 0x94a3b8, 2.7);
  scene.add(hemi);

  const sun = new THREE.DirectionalLight(0xffffff, 2.4);
  sun.position.set(cols * 0.4, Math.max(rows, cols) * 1.1, -rows * 0.45);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.left = -cols;
  sun.shadow.camera.right = cols;
  sun.shadow.camera.top = rows;
  sun.shadow.camera.bottom = -rows;
  scene.add(sun);
}

function buildLayout() {
  const { rows, cols } = trace.layout.grid_size;
  const baseGeometry = new THREE.BoxGeometry(cols * TILE + 1.6, 0.08, rows * TILE + 1.6);
  const baseMaterial = new THREE.MeshStandardMaterial({
    color: COLORS.blocked,
    roughness: 0.92,
    metalness: 0.02,
  });
  const base = new THREE.Mesh(baseGeometry, baseMaterial);
  base.position.y = -0.08;
  base.receiveShadow = true;
  scene.add(base);

  const roadMaterial = material(COLORS.road);
  const startMaterial = material(COLORS.start);
  const exitMaterial = material(COLORS.exit);
  const waitingMaterial = material(COLORS.waiting);
  const conflictMaterial = material(COLORS.conflict);
  const tileGeometry = new THREE.BoxGeometry(0.94, 0.08, 0.94);

  for (const node of trace.layout.nodes) {
    const mesh = new THREE.Mesh(tileGeometry, materialForNode(node, {
      road: roadMaterial,
      start: startMaterial,
      exit: exitMaterial,
      waiting: waitingMaterial,
      conflict: conflictMaterial,
    }));
    const p = gridToWorld(node.row, node.col, 0);
    mesh.position.set(p.x, 0, p.z);
    mesh.receiveShadow = true;
    mesh.userData.nodeId = node.id;
    scene.add(mesh);
  }

  buildLaneLines();
  buildSharedPulse();
}

function buildLaneLines() {
  const points = [];
  for (const edge of trace.layout.edges) {
    const from = nodeById.get(edge.from);
    const to = nodeById.get(edge.to);
    if (!from || !to) continue;
    const a = gridToWorld(from.row, from.col, 0.055);
    const b = gridToWorld(to.row, to.col, 0.055);
    points.push(new THREE.Vector3(a.x, a.y, a.z), new THREE.Vector3(b.x, b.y, b.z));
  }
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const line = new THREE.LineSegments(
    geometry,
    new THREE.LineBasicMaterial({
      color: COLORS.laneLine,
      transparent: true,
      opacity: 0.34,
    }),
  );
  scene.add(line);
}

function buildSharedPulse() {
  for (const nodeId of trace.layout.interaction_area.conflict_zone_nodes) {
    const node = nodeById.get(nodeId);
    if (!node) continue;
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.38, 0.47, 48),
      new THREE.MeshBasicMaterial({
        color: COLORS.sharedRing,
        transparent: true,
        opacity: 0.25,
        side: THREE.DoubleSide,
      }),
    );
    const p = gridToWorld(node.row, node.col, 0.095);
    ring.position.set(p.x, p.y, p.z);
    ring.rotation.x = -Math.PI / 2;
    scene.add(ring);
    sharedPulse.push(ring);
  }
}

function buildRobots() {
  for (const robotId of allRobotIds) {
    const group = makeRobotMesh(robotId);
    group.visible = false;
    robotMeshes.set(robotId, group);
    scene.add(group);
  }
}

function makeRobotMesh(robotId) {
  const sample = findRobotSample(robotId);
  const color = new THREE.Color(sample?.color || "#4f83c4");
  const group = new THREE.Group();
  group.userData.target = new THREE.Vector3();
  group.userData.heading = "SOUTH";

  const body = new THREE.Mesh(
    new THREE.BoxGeometry(0.62, 0.28, 0.78),
    new THREE.MeshStandardMaterial({
      color,
      roughness: 0.42,
      metalness: 0.18,
    }),
  );
  body.position.y = ROBOT_HEIGHT;
  body.castShadow = true;
  group.add(body);

  const deck = new THREE.Mesh(
    new THREE.BoxGeometry(0.44, 0.12, 0.48),
    new THREE.MeshStandardMaterial({
      color: color.clone().offsetHSL(0, -0.05, 0.16),
      roughness: 0.38,
      metalness: 0.24,
    }),
  );
  deck.position.y = ROBOT_HEIGHT + 0.2;
  deck.castShadow = true;
  group.add(deck);

  const arrow = new THREE.Mesh(
    new THREE.ConeGeometry(0.13, 0.38, 28),
    new THREE.MeshStandardMaterial({
      color: 0xffffff,
      roughness: 0.3,
      metalness: 0.04,
    }),
  );
  arrow.rotation.x = Math.PI / 2;
  arrow.position.set(0, ROBOT_HEIGHT + 0.21, 0.18);
  arrow.castShadow = true;
  group.add(arrow);

  const wheelMaterial = new THREE.MeshStandardMaterial({ color: 0x18212b, roughness: 0.58 });
  for (const x of [-0.38, 0.38]) {
    for (const z of [-0.26, 0.26]) {
      const wheel = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, 0.08, 18), wheelMaterial);
      wheel.rotation.z = Math.PI / 2;
      wheel.position.set(x, ROBOT_HEIGHT - 0.1, z);
      wheel.castShadow = true;
      group.add(wheel);
    }
  }

  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(0.48, 0.025, 10, 52),
    new THREE.MeshBasicMaterial({ color: 0x111827, transparent: true, opacity: 0 }),
  );
  ring.rotation.x = Math.PI / 2;
  ring.position.y = 0.08;
  group.userData.sharedRing = ring;
  group.add(ring);

  return group;
}

function bindControls() {
  els.playButton.addEventListener("click", togglePlay);
  els.prevButton.addEventListener("click", () => goToFrame(Math.max(0, frameIndex - 1)));
  els.nextButton.addEventListener("click", () => goToFrame(Math.min(trace.frames.length - 1, frameIndex + 1)));
  els.resetButton.addEventListener("click", () => goToFrame(0));
  els.topButton.addEventListener("click", setTopCamera);
  els.orbitButton.addEventListener("click", setOrbitCamera);
  els.stepSlider.addEventListener("input", () => goToFrame(Number(els.stepSlider.value)));
  els.speedSlider.addEventListener("input", updateSpeedOutput);
  els.robotSelect.addEventListener("change", () => {
    selectedRobotId = els.robotSelect.value;
    updatePathLine();
  });
}

function configureUi() {
  const max = Math.max(0, trace.frames.length - 1);
  els.stepSlider.max = String(max);
  els.stepSlider.value = "0";
  els.robotSelect.innerHTML = "";
  for (const robotId of allRobotIds) {
    const option = document.createElement("option");
    option.value = robotId;
    option.textContent = robotId;
    els.robotSelect.append(option);
  }
  els.robotSelect.value = selectedRobotId;

  const meta = trace.metadata;
  els.runLabel.textContent = [
    trace.layout.name,
    `robots/dir ${meta.robots_per_direction}`,
    `capacity ${meta.shared_area_capacity}`,
    `seed ${meta.seed}`,
  ].join(" | ");
  els.policyStat.textContent = `Policy ${meta.policy_type}`;
  updateSpeedOutput();
}

function buildLegend() {
  const items = [
    ["Road", "#f9fbfd"],
    ["Start", "#86d59a"],
    ["Exit", "#527fc2"],
    ["Waiting", "#f0c45c"],
    ["Conflict", "#e58e70"],
  ];
  els.legend.innerHTML = "";
  for (const [label, color] of items) {
    const item = document.createElement("span");
    item.className = "legend-item";
    item.innerHTML = `<span class="swatch" style="background:${color}"></span><span>${label}</span>`;
    els.legend.append(item);
  }
}

function togglePlay() {
  playing = !playing;
  els.playButton.textContent = playing ? "Pause" : "Play";
  els.playButton.classList.toggle("is-active", playing);
  lastTick = performance.now();
}

function goToFrame(index) {
  frameIndex = Math.max(0, Math.min(trace.frames.length - 1, index));
  const frame = trace.frames[frameIndex];
  els.stepSlider.value = String(frameIndex);
  els.stepOutput.textContent = `${frame.step} / ${trace.frames[trace.frames.length - 1].step}`;
  els.sharedStat.textContent = `Shared ${formatList(frame.shared_robot_ids)}`;
  els.queueStat.textContent = `Queue ${frame.queue.length}`;
  els.doneStat.textContent = `Done ${frame.completed_count}/${trace.metrics.robots}`;
  updateRobots(frame);
  updatePathLine();
  if (frameIndex === trace.frames.length - 1 && playing) {
    togglePlay();
  }
}

function updateRobots(frame) {
  const visibleIds = new Set();
  for (const robot of frame.robots) {
    const mesh = robotMeshes.get(robot.id);
    if (!mesh || robot.row == null || robot.col == null) continue;
    const p = gridToWorld(robot.row, robot.col, ROBOT_HEIGHT);
    mesh.userData.target.set(p.x, p.y, p.z);
    if (!mesh.visible) {
      mesh.position.copy(mesh.userData.target);
    }
    mesh.rotation.y = headingAngle(robot.heading);
    mesh.visible = true;
    mesh.userData.heading = robot.heading;
    mesh.userData.status = robot.status;
    mesh.userData.sharedRing.material.opacity = robot.is_shared ? 0.75 : 0;
    mesh.traverse((child) => {
      if (child.material && child !== mesh.userData.sharedRing) {
        child.material.opacity = robot.status === "completed" ? 0.52 : 1;
        child.material.transparent = robot.status === "completed";
      }
    });
    visibleIds.add(robot.id);
  }
  for (const [robotId, mesh] of robotMeshes.entries()) {
    if (!visibleIds.has(robotId)) {
      mesh.visible = false;
    }
  }
}

function updatePathLine() {
  if (pathLine) {
    scene.remove(pathLine);
    pathLine.geometry.dispose();
    pathLine.material.dispose();
    pathLine = null;
  }
  const robot = getFrameRobot(selectedRobotId);
  if (!robot || !robot.path) return;

  const points = [];
  for (const nodeId of robot.path) {
    const node = nodeById.get(nodeId);
    if (!node) continue;
    const p = gridToWorld(node.row, node.col, 0.17);
    points.push(new THREE.Vector3(p.x, p.y, p.z));
  }
  if (points.length < 2) return;
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  pathLine = new THREE.Line(
    geometry,
    new THREE.LineBasicMaterial({
      color: new THREE.Color(robot.color || "#1f6feb"),
      linewidth: 2,
      transparent: true,
      opacity: 0.85,
    }),
  );
  scene.add(pathLine);
}

function animate(time) {
  requestAnimationFrame(animate);
  controls.update();

  if (playing) {
    const interval = 1000 / Number(els.speedSlider.value);
    if (time - lastTick >= interval) {
      const elapsedSteps = Math.max(1, Math.floor((time - lastTick) / interval));
      lastTick = time;
      goToFrame(Math.min(trace.frames.length - 1, frameIndex + elapsedSteps));
    }
  }

  for (const mesh of robotMeshes.values()) {
    if (!mesh.visible) continue;
    mesh.position.lerp(mesh.userData.target, 0.2);
  }

  const pulse = 0.22 + Math.sin(time * 0.004) * 0.08;
  for (const ring of sharedPulse) {
    ring.material.opacity = pulse;
    const scale = 1 + Math.sin(time * 0.004) * 0.08;
    ring.scale.set(scale, scale, scale);
  }

  if (selectedRobotId && controls.enabled) {
    const selected = robotMeshes.get(selectedRobotId);
    if (selected?.visible && document.activeElement !== els.stepSlider) {
      controls.target.lerp(selected.position, 0.015);
    }
  }

  renderer.render(scene, camera);
}

function setTopCamera() {
  const { rows } = trace.layout.grid_size;
  camera.position.set(0, Math.max(rows, 10) * 1.35, 0.001);
  controls.target.set(0, 0, 0);
  controls.update();
}

function setOrbitCamera() {
  const { rows, cols } = trace.layout.grid_size;
  camera.position.set(cols * 0.7, Math.max(rows, cols) * 0.9, rows * 0.8);
  controls.target.set(0, 0, 0);
  controls.update();
}

function resizeRenderer() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

function updateSpeedOutput() {
  els.speedOutput.textContent = `${els.speedSlider.value} steps/s`;
}

function collectRobotIds(frames) {
  const ids = new Set();
  for (const frame of frames) {
    for (const robot of frame.robots) {
      ids.add(robot.id);
    }
  }
  return [...ids].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
}

function findRobotSample(robotId) {
  for (const frame of trace.frames) {
    const robot = frame.robots.find((item) => item.id === robotId);
    if (robot) return robot;
  }
  return null;
}

function getFrameRobot(robotId) {
  const frame = trace.frames[frameIndex];
  return frame.robots.find((item) => item.id === robotId) || findRobotSample(robotId);
}

function formatList(items) {
  return items?.length ? items.join(",") : "-";
}

function material(color) {
  return new THREE.MeshStandardMaterial({
    color,
    roughness: 0.82,
    metalness: 0.02,
  });
}

function materialForNode(node, materials) {
  if (node.role === "start") return materials.start;
  if (node.role === "exit") return materials.exit;
  if (node.role === "waiting") return materials.waiting;
  if (node.role === "conflict") return materials.conflict;
  return materials.road;
}

function gridToWorld(row, col, y = 0) {
  const { rows, cols } = trace.layout.grid_size;
  return {
    x: (col - (cols - 1) / 2) * TILE,
    y,
    z: (row - (rows - 1) / 2) * TILE,
  };
}

function headingAngle(heading) {
  return {
    NORTH: Math.PI,
    SOUTH: 0,
    EAST: Math.PI / 2,
    WEST: -Math.PI / 2,
  }[heading] ?? 0;
}
