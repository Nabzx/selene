/**
 * SELENE — Hard-Coded Live Demo UI Controller (Concept Panels 02–08)
 *
 * Implements the full 8-stage storyboard flow with deterministic, zero-crash
 * hard-coded simulation data from GitHub Run 08 (500k episodes / ~18M steps):
 *   1-2. Rotating 3D Moon + S-band telemetry ellipse + Drag & Drop .TXT
 *   3.   Zoom into Daedalus Top-Down Map View ([DATA], [INIT SIM])
 *   4-5. Multi-hop 4-hopper path planning + Horizontal Training Iteration
 *        Scrollbar (1,000 -> 10,000 -> 100,000 -> 500k episodes / 18M steps)
 *   6.   [EXECUTE / AUTHORISE UPLINK] -> Follow-cam autonomous multi-hop flight
 *   7.   [LOCAL SIM OVERRIDE] -> Free 6-DOF WASD/Space/Shift control (no
 *        horizon lock, can flip upside down, crash & [RESPAWN HOPPER])
 *   8.   3-Pane Mission Summary & [EXPORT] screen (OVERVIEW / SUMMARY / LOGS)
 */

import { SeleneScene } from "./scene.js";

const SAMPLE_MISSION_TXT = `# SELENE FAR-SIDE AUTONOMOUS HOPPER MISSION SPECIFICATION
# ARCHIVAL MAG 44 // FRAME 6842 · CESIUM MOON + NASA LRO SLDEM2015
MISSION_ID: SELENE-DAEDALUS-RUN08
SITE: DAEDALUS_CRATER_FAR_SIDE
CENTER_LAT_LON: -5.90, 179.40
GRAVITY_MPS2: 1.62
FLEET_SIZE: 4

# MULTI-HOP WAYPOINTS (START -> CRATER HOPS -> END)
START_COORD: c1 (-1950, +520) [SLOPE 4.2 DEG]
HOP_WAYPOINT_1: c2 (-480, +310) [SCIENCE TARGET ALPHA]
HOP_WAYPOINT_2: c3 (+560, -140) [SCIENCE TARGET BETA]
END_COORD: c4 (+1920, +60) [L2 RELAY EXTRACTION ZONE]

# GEMINI 4 ARGON SAFETY VERIFIER CONSTRAINTS (1,000 MONTE CARLO ROLLOUTS)
RULE_R1: BATTERY_MIN_PCT >= 18.0
RULE_R2: FUEL_RESERVE_MIN_PCT >= 10.0
RULE_R3: TOUCHDOWN_SLOPE_MAX_DEG <= 15.0
RULE_R4: THERMAL_SHELL_MAX_K <= 380.0
RULE_R5: TDMA_RELAY_CONFLICT_MAX <= 0.02

# CTDE MAPPO + 5-MLP ENSEMBLE WORLD MODEL CHECKPOINT
TRAINING_EPISODES: 500000
TRAINING_STEPS: 18000000
CONVERGED_REWARD: 539.88
WORLD_MODEL_MSE: 0.0137
`;

// Instantiate 3D Scene
const viewportEl = document.getElementById("viewport");
const scene = new SeleneScene(viewportEl);

// UI State
let currentStage = "HOME"; // HOME | MAP_TOP | SIM_PATHS | FLIGHT | SUMMARY
let selectedHopperId = 0; // 0 = All Fleet, 1..4 = Hopper 1..4
let autoScrubTimer = null;
let overviewBgImg = null;

// Preload top-down crater concept image for Stage 8 Overview pane
const preloadImg = new Image();
preloadImg.onload = () => {
  overviewBgImg = preloadImg;
  if (currentStage === "SUMMARY") drawSummaryOverviewCanvas();
};
preloadImg.src = "/data/concept/panel03-a.jpg";

// DOM Elements
const stageHomeOverlay = document.getElementById("stage-home-overlay");
const stageMapBar = document.getElementById("stage-map-bar");
const dataDrawer = document.getElementById("data-drawer");
const violationBox = document.getElementById("violation-callout-box");
const violationTitle = document.getElementById("violation-title");
const violationSub = document.getElementById("violation-sub");
const stageSimDock = document.getElementById("stage-sim-dock");
const stageFlightHud = document.getElementById("stage-flight-hud");
const stageSummaryModal = document.getElementById("stage-summary-modal");
const miniMoonWidget = document.getElementById("mini-moon-widget");
const topRightTools = document.getElementById("top-right-tools");
const brandSignature = document.getElementById("brand-signature");
const stampTl = document.getElementById("stamp-tl");
const stampTr = document.getElementById("stamp-tr");
const toastEl = document.getElementById("archival-toast");

// Scrubber DOM
const scrubberInput = document.getElementById("training-scrubber");
const scrubberIterLabel = document.getElementById("scrubber-iter-label");
const scrubberStageDesc = document.getElementById("scrubber-stage-desc");
const metricReward = document.getElementById("metric-reward");
const metricFailed = document.getElementById("metric-failed");
const metricGreen = document.getElementById("metric-green");
const metricMse = document.getElementById("metric-mse");
const metricCrash = document.getElementById("metric-crash");

// Flight & Manual Override DOM
const btnManualOverride = document.getElementById("btn-manual-override");
const btnRespawnTop = document.getElementById("btn-respawn-top");
const btnRespawnModal = document.getElementById("btn-respawn-modal");
const manualCrashBanner = document.getElementById("manual-crash-banner");
const manualCrashReason = document.getElementById("manual-crash-reason");
const flightGravityCallout = document.getElementById("flight-gravity-callout");

// Toast helper
let toastTimeout = null;
function showToast(msg, durationMs = 2400) {
  if (!toastEl) return;
  toastEl.textContent = msg;
  toastEl.classList.add("visible");
  if (toastTimeout) clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => {
    toastEl.classList.remove("visible");
  }, durationMs);
}

// ============================================================================
// STAGE TRANSITIONS
// ============================================================================
function setStage(stage) {
  currentStage = stage;
  stopAutoScrub();

  // Hide all stage overlays first
  stageHomeOverlay.classList.add("hidden");
  stageMapBar.classList.add("hidden");
  dataDrawer.classList.add("hidden");
  violationBox.classList.add("hidden");
  stageSimDock.classList.add("hidden");
  stageFlightHud.classList.add("hidden");
  stageSummaryModal.classList.add("hidden");

  if (stage === "HOME") {
    stageHomeOverlay.classList.remove("hidden");
    miniMoonWidget.classList.add("hidden");
    topRightTools.classList.add("hidden");
    brandSignature.style.opacity = "1";
    stampTl.textContent = "MAG 44 // FRAME 6842 · 70MM RAW ARCHIVAL";
    stampTr.textContent = "LUNAR FAR SIDE · 05.9°S 179.4°E";
    scene.transitionToGlobe();
  } else if (stage === "MAP_TOP") {
    stageMapBar.classList.remove("hidden");
    miniMoonWidget.classList.remove("hidden");
    topRightTools.classList.remove("hidden");
    brandSignature.style.opacity = "0.85";
    stampTl.textContent = "DAEDALUS · ORTHOGRAPHIC DEM · SLDEM2015";
    stampTr.textContent = "START c1 → c2 → c3 → END c4";
  } else if (stage === "SIM_PATHS") {
    violationBox.classList.remove("hidden");
    stageSimDock.classList.remove("hidden");
    miniMoonWidget.classList.remove("hidden");
    topRightTools.classList.remove("hidden");
    brandSignature.style.opacity = "0";
    stampTl.textContent = "CTDE MAPPO + 5-MLP WORLD MODEL IMAGINATION";
    stampTr.textContent = "4 HOPPERS · MULTI-HOP TRAJECTORY OPTIMISATION";
  } else if (stage === "FLIGHT") {
    stageFlightHud.classList.remove("hidden");
    miniMoonWidget.classList.remove("hidden");
    topRightTools.classList.remove("hidden");
    brandSignature.style.opacity = "0";
    stampTl.textContent = "LIVE TELEMETRY · 1.62 m/s² BALLISTIC FLIGHT";
    stampTr.textContent = "COPPER-TANK PROPULSIVE HOPPER · GIMBAL LOCKED";
  } else if (stage === "SUMMARY") {
    stageSummaryModal.classList.remove("hidden");
    miniMoonWidget.classList.add("hidden");
    topRightTools.classList.add("hidden");
    brandSignature.style.opacity = "0";
    stampTl.textContent = "MISSION DEBRIEF // ARGON VERIFIER PASSED";
    stampTr.textContent = "DAEDALUS · 12 CRATERS MAPPED · 0 CRASHES";
    drawSummaryOverviewCanvas();
  }
}

// ============================================================================
// STAGE 1 & 2 -> STAGE 3: MISSION .TXT UPLOAD / DRAG & DROP
// ============================================================================
function loadMissionAndZoomToTopMap(sourceLabel = "DAEDALUS_MISSION_SPEC.TXT") {
  showToast(`LOADED ${sourceLabel} — ZOOMING TO DAEDALUS CRATER`);
  setStage("MAP_TOP");
  scene.zoomFromGlobeToSurfaceTopView(() => {
    showToast("TOP-DOWN MAP LOCKED — CLICK [INIT SIM] FOR PATH PLANNING");
  });
}

document.getElementById("btn-load-default-mission").addEventListener("click", () => {
  loadMissionAndZoomToTopMap("DAEDALUS_MISSION_SPEC.TXT");
});

document.getElementById("btn-browse-mission").addEventListener("click", () => {
  document.getElementById("mission-file-input").click();
});

document.getElementById("mission-file-input").addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  if (file) {
    loadMissionAndZoomToTopMap(file.name.toUpperCase());
  }
});

document.getElementById("btn-download-sample-txt").addEventListener("click", () => {
  const blob = new Blob([SAMPLE_MISSION_TXT], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "DAEDALUS_MISSION_SPEC.txt";
  a.click();
  URL.revokeObjectURL(url);
  showToast("DOWNLOADED DAEDALUS_MISSION_SPEC.TXT");
});

// Drag & Drop anywhere on window or dock
const homeDropDock = document.getElementById("home-drop-dock");
window.addEventListener("dragover", (e) => {
  e.preventDefault();
  if (homeDropDock) homeDropDock.classList.add("dragover");
});
window.addEventListener("dragleave", () => {
  if (homeDropDock) homeDropDock.classList.remove("dragover");
});
window.addEventListener("drop", (e) => {
  e.preventDefault();
  if (homeDropDock) homeDropDock.classList.remove("dragover");
  const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
  const name = file ? file.name.toUpperCase() : "DAEDALUS_MISSION_SPEC.TXT";
  loadMissionAndZoomToTopMap(name);
});

// ============================================================================
// STAGE 3: TOP-DOWN MAP ([DATA] & [INIT SIM])
// ============================================================================
document.getElementById("btn-toggle-data").addEventListener("click", () => {
  dataDrawer.classList.toggle("hidden");
});
document.getElementById("btn-close-data").addEventListener("click", () => {
  dataDrawer.classList.add("hidden");
});

document.getElementById("btn-init-sim").addEventListener("click", () => {
  setStage("SIM_PATHS");
  scene.tiltToSimulationView();
  // Start scrubber at 1,000 steps (early exploration with red failed paths)
  updateTrainingScrubber(60);
  showToast("ROUTE SIMULATION INITIALISED — SCRUB BAR TO SEE RL CONVERGENCE");
});

// Top-right camera helper buttons
document.getElementById("btn-top-view").addEventListener("click", () => {
  scene.setManualMode(false);
  scene.snapToTopMapView();
  if (currentStage === "FLIGHT") setStage("MAP_TOP");
});
document.getElementById("btn-tilt-view").addEventListener("click", () => {
  scene.setManualMode(false);
  setStage("SIM_PATHS");
  scene.tiltToSimulationView();
  updateTrainingScrubber(Number(scrubberInput.value));
});
document.getElementById("btn-globe-view").addEventListener("click", () => {
  scene.setManualMode(false);
  setStage("HOME");
});
document.getElementById("btn-open-summary").addEventListener("click", () => {
  scene.setManualMode(false);
  setStage("SUMMARY");
});

// ============================================================================
// STAGE 4 & 5: HORIZONTAL TRAINING ITERATION SCRUBBER (1K -> 10K -> 100K -> 500K/18M)
// ============================================================================
function updateTrainingScrubber(rawVal) {
  const clamped = Math.max(0, Math.min(1000, Number(rawVal)));
  scrubberInput.value = String(clamped);
  const scrub01 = clamped / 1000;

  const info = scene.setTrainingConvergenceScrub(scrub01);

  // Format human-readable iteration & step count matching user's prompt + Run 08
  let iterText = "";
  let descText = "";
  if (scrub01 < 0.22) {
    const steps = Math.round(1000 + (scrub01 / 0.22) * 4000);
    const eps = Math.round(steps / 36);
    iterText = `ITERATION: ${steps.toLocaleString()} STEPS (~${eps.toLocaleString()} EPISODES)`;
    descText = "EARLY WORLD-MODEL EXPLORATION · RANDOM BALLISTIC RED CRASHES";
  } else if (scrub01 < 0.58) {
    const t = (scrub01 - 0.22) / 0.36;
    const steps = Math.round(10000 + t * 65000);
    const eps = Math.round(steps / 36);
    iterText = `ITERATION: ${steps.toLocaleString()} STEPS (~${eps.toLocaleString()} EPISODES)`;
    descText = "10K+ STEP REGIME · GREEN CANDIDATE MULTI-HOP SPREAD EMERGING";
  } else if (scrub01 < 0.90) {
    const t = (scrub01 - 0.58) / 0.32;
    const steps = Math.round(100000 + t * 4900000);
    const eps = Math.round(steps / 36);
    iterText = `ITERATION: ${steps.toLocaleString()} STEPS (~${eps.toLocaleString()} EPISODES)`;
    descText = "100K+ STEP REGIME · PRUNING R3 SLOPE VIOLATIONS · CONVERGING";
  } else {
    iterText = "CONVERGED: 500,000 EPISODES / 18,000,000 STEPS (RUN 08)";
    descText = "OPTIMAL 4-HOPPER MULTI-HOP ROUTES LOCKED · 100% ARGON VERIFIED";
  }

  scrubberIterLabel.textContent = iterText;
  scrubberStageDesc.textContent = descText;

  // Update Metrics Strip
  metricReward.textContent = info.reward.toFixed(2);
  metricReward.className = "v " + (scrub01 > 0.7 ? "green" : scrub01 > 0.3 ? "amber" : "red");

  metricFailed.textContent = `${info.redVisible} RED`;
  metricGreen.textContent = `${info.greenVisible + 4} GREEN`;
  metricMse.textContent = info.wmMse.toFixed(4);

  const crashPct = (info.crashRate * 100).toFixed(1) + "%";
  metricCrash.textContent = crashPct;
  metricCrash.className = "v " + (info.crashRate < 0.02 ? "green" : info.crashRate < 0.25 ? "amber" : "red");

  // Update off-white Violation Callout Box (panel04-b.jpg -> panel05-d.jpg)
  if (scrub01 < 0.45) {
    violationBox.classList.remove("converged");
    violationTitle.textContent = "R3 VIOLATION · CRASH DETECTED";
    violationSub.innerHTML = `TOUCHDOWN SLOPE 21.4 DEG &gt; 15.0 DEG<br/>${info.redVisible} / 128 ROLLOUTS TERMINATED ON CRATER RIM`;
  } else if (scrub01 < 0.88) {
    violationBox.classList.remove("converged");
    violationTitle.textContent = "ARGON VERIFIER · PRUNING UNSAFE ARCS";
    violationSub.innerHTML = `${info.greenVisible + 4} VIABLE MULTI-HOP PATHS FOUND<br/>${info.redVisible} BOUNDARY ROLLOUTS REJECTED (R3 SLOPE)`;
  } else {
    violationBox.classList.add("converged");
    violationTitle.textContent = "ARGON VERIFIER · 1,000 / 1,000 PASS";
    violationSub.innerHTML = `ALL 4 HOPPERS LOCKED ON OPTIMAL GREEN PATHS<br/>SLOPE MAX 6.8° &lt; 15.0° · TDMA CONFLICT 0.0156`;
  }

  // Highlight matching preset tick button
  document.querySelectorAll(".scrubber-tick-btn").forEach((btn) => {
    const target = Number(btn.getAttribute("data-scrub"));
    btn.classList.toggle("active", Math.abs(clamped - target) < 95);
  });
}

scrubberInput.addEventListener("input", (e) => {
  stopAutoScrub();
  updateTrainingScrubber(e.target.value);
});

document.querySelectorAll(".scrubber-tick-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    stopAutoScrub();
    const val = Number(btn.getAttribute("data-scrub"));
    updateTrainingScrubber(val);
  });
});

// Auto-converge demo button (smoothly animates scrollbar from 1,000 steps -> 500k ep / 18M steps)
function stopAutoScrub() {
  if (autoScrubTimer) {
    clearInterval(autoScrubTimer);
    autoScrubTimer = null;
    const btn = document.getElementById("btn-auto-scrub");
    if (btn) btn.textContent = "[AUTO-CONVERGE]";
  }
}

document.getElementById("btn-auto-scrub").addEventListener("click", () => {
  if (autoScrubTimer) {
    stopAutoScrub();
    return;
  }
  const btn = document.getElementById("btn-auto-scrub");
  btn.textContent = "[PAUSE SCRUB]";
  if (Number(scrubberInput.value) >= 980) {
    scrubberInput.value = "20";
  }
  autoScrubTimer = setInterval(() => {
    const next = Number(scrubberInput.value) + 8;
    if (next >= 1000) {
      updateTrainingScrubber(1000);
      stopAutoScrub();
      showToast("CONVERGED AT 500K EPISODES / 18M STEPS — CLICK [EXECUTE]");
    } else {
      updateTrainingScrubber(next);
    }
  }, 35);
});

// Individual Hopper Selector Tabs (Zoom in on Hopper 1..4 or All Fleet)
document.querySelectorAll(".hopper-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".hopper-tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    selectedHopperId = Number(tab.getAttribute("data-hopper"));
    scene.selectHopper(selectedHopperId);
    if (selectedHopperId === 0) {
      showToast("VIEWING ALL 4 FLEET HOPPERS");
    } else {
      showToast(`ZOOMED IN ON HOPPER ${selectedHopperId} MULTI-HOP PATH`);
    }
  });
});

// ============================================================================
// STAGE 6: [EXECUTE / AUTHORISE UPLINK] -> AUTONOMOUS FOLLOW-CAM FLIGHT
// ============================================================================
document.getElementById("btn-execute-sim").addEventListener("click", () => {
  // Snap scrubber to 100% converged green paths when executing mission
  updateTrainingScrubber(1000);
  setStage("FLIGHT");
  manualCrashBanner.classList.add("hidden");
  btnRespawnTop.classList.add("hidden");
  btnManualOverride.textContent = "[LOCAL SIM OVERRIDE]";
  btnManualOverride.classList.add("amber");
  btnManualOverride.classList.remove("primary");
  flightGravityCallout.textContent = "1.62 m/s² · AUTONOMOUS FOLLOW";

  const targetHopper = selectedHopperId > 0 ? selectedHopperId : 1;
  scene.startExecutedFlight(targetHopper);
  showToast(`EXECUTING HOPPER ${targetHopper} MULTI-HOP MISSION (c1 → c2 → c3 → c4)`);
});

document.getElementById("btn-back-to-paths").addEventListener("click", () => {
  scene.setManualMode(false);
  setStage("SIM_PATHS");
  scene.tiltToSimulationView();
});

document.getElementById("btn-complete-mission").addEventListener("click", () => {
  scene.setManualMode(false);
  setStage("SUMMARY");
  showToast("MISSION TERMINATED — DISPLAYING 3-PANE SUMMARY & LOGS");
});

// ============================================================================
// STAGE 7: [LOCAL SIM OVERRIDE] -> FREE 6-DOF WASD/SPACE/SHIFT + [RESPAWN]
// ============================================================================
function activateManual6DofMode() {
  scene.setManualMode(true);
  manualCrashBanner.classList.add("hidden");
  btnRespawnTop.classList.remove("hidden");
  btnManualOverride.textContent = "[RETURN TO AUTO AI]";
  btnManualOverride.classList.remove("amber");
  btnManualOverride.classList.add("primary");
  flightGravityCallout.textContent = "1.62 m/s² · UNCONSTRAINED 6-DOF MANUAL";
  showToast("FREE 6-DOF MANUAL OVERRIDE — WASD FLIPS ANY ORIENTATION, SPACE THRUSTS, R RESPAWNS");
}

function deactivateManual6DofMode() {
  scene.setManualMode(false);
  manualCrashBanner.classList.add("hidden");
  btnRespawnTop.classList.add("hidden");
  btnManualOverride.textContent = "[LOCAL SIM OVERRIDE]";
  btnManualOverride.classList.add("amber");
  btnManualOverride.classList.remove("primary");
  flightGravityCallout.textContent = "1.62 m/s² · AUTONOMOUS FOLLOW";
  scene.startExecutedFlight(selectedHopperId > 0 ? selectedHopperId : 1);
  showToast("RETURNED TO AUTONOMOUS MAPPO FLIGHT");
}

btnManualOverride.addEventListener("click", () => {
  if (!scene.manualMode) {
    activateManual6DofMode();
  } else {
    deactivateManual6DofMode();
  }
});

function triggerRespawn() {
  scene.respawnManualHopper();
  manualCrashBanner.classList.add("hidden");
  showToast("HOPPER RESPAWNED UPRIGHT ABOVE DAEDALUS FLOOR");
}

btnRespawnTop.addEventListener("click", triggerRespawn);
btnRespawnModal.addEventListener("click", triggerRespawn);

// Wire Crash & Respawn callbacks from scene.js
scene.onManualCrash = (info) => {
  manualCrashReason.textContent = `${info.reason} — IMPACT SPEED ${info.speed} m/s · TILT ${info.tiltDeg}° (NO HORIZON LOCK)`;
  manualCrashBanner.classList.remove("hidden");
  showToast("CRASH DETECTED! CLICK [RESPAWN HOPPER] OR PRESS R");
};

scene.onManualRespawn = () => {
  manualCrashBanner.classList.add("hidden");
};

scene.onFlightFinished = () => {
  if (currentStage === "FLIGHT" && !scene.manualMode) {
    showToast("ALL HOPS COMPLETED (c1 → c2 → c3 → c4) — OPENING MISSION SUMMARY");
    setTimeout(() => {
      if (currentStage === "FLIGHT" && !scene.manualMode) {
        setStage("SUMMARY");
      }
    }, 1200);
  }
};

// Keyboard listeners for WASD / Space / Shift + R respawn
const keyMap = {
  KeyW: { prop: "w", el: document.getElementById("key-w") },
  KeyS: { prop: "s", el: document.getElementById("key-s") },
  KeyA: { prop: "a", el: document.getElementById("key-a") },
  KeyD: { prop: "d", el: document.getElementById("key-d") },
  Space: { prop: "space", el: document.getElementById("key-space") },
  ShiftLeft: { prop: "shift", el: document.getElementById("key-shift") },
  ShiftRight: { prop: "shift", el: document.getElementById("key-shift") },
};

window.addEventListener("keydown", (e) => {
  if (e.code === "KeyR" && currentStage === "FLIGHT" && scene.manualMode) {
    e.preventDefault();
    triggerRespawn();
    return;
  }
  const entry = keyMap[e.code];
  if (entry && currentStage === "FLIGHT") {
    if (e.code === "Space") e.preventDefault();
    // Auto-enable manual override if user presses flight keys during Stage 6
    if (!scene.manualMode) {
      activateManual6DofMode();
    }
    scene.setManualKey(entry.prop, true);
    if (entry.el) entry.el.classList.add("active");
  }
});

window.addEventListener("keyup", (e) => {
  const entry = keyMap[e.code];
  if (entry) {
    scene.setManualKey(entry.prop, false);
    if (entry.el) entry.el.classList.remove("active");
  }
});

// ============================================================================
// STAGE 8: 3-PANE MISSION SUMMARY & [EXPORT] (panel08-a.jpg)
// ============================================================================
function drawSummaryOverviewCanvas() {
  const canvas = document.getElementById("summary-overview-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = "#07090c";
  ctx.fillRect(0, 0, w, h);

  // Draw panel03-a crater background if loaded
  if (overviewBgImg) {
    ctx.save();
    ctx.globalAlpha = 0.72;
    ctx.drawImage(overviewBgImg, 0, 0, w, h);
    ctx.restore();
  }

  // Subtle archival grid
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth = 1;
  for (let x = 90; x < w; x += 90) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 90; y < h; y += 90) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Draw crater circles c1..c4
  const craters = [
    { id: "START c1", x: w * 0.16, y: h * 0.68, r: 34 },
    { id: "HOP c2", x: w * 0.41, y: h * 0.52, r: 52 },
    { id: "HOP c3", x: w * 0.63, y: h * 0.44, r: 44 },
    { id: "END c4", x: w * 0.84, y: h * 0.34, r: 38 },
  ];

  craters.forEach((c) => {
    ctx.strokeStyle = "rgba(255,255,255,0.45)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(c.x, c.y, c.r, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = "#eef2f6";
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.fillText(c.id, c.x - 26, c.y + c.r + 16);
  });

  // Draw the 4 executed green multi-hop trajectories across c1 -> c2 -> c3 -> c4
  const offsets = [-22, -7, 8, 23];
  offsets.forEach((off, idx) => {
    ctx.strokeStyle = idx === 0 ? "#4fffa3" : "#3ddc84";
    ctx.lineWidth = idx === 0 ? 2.6 : 1.8;
    ctx.beginPath();
    for (let hop = 0; hop < craters.length - 1; hop++) {
      const p0 = craters[hop];
      const p1 = craters[hop + 1];
      const mx = (p0.x + p1.x) * 0.5;
      const my = Math.min(p0.y, p1.y) - 48 + off * 1.4;
      if (hop === 0) ctx.moveTo(p0.x, p0.y + off);
      ctx.quadraticCurveTo(mx, my, p1.x, p1.y + off * 0.6);
    }
    ctx.stroke();
  });
}

document.getElementById("btn-export-drone").addEventListener("click", () => {
  const payload = {
    mission_id: "SELENE-DAEDALUS-RUN08",
    timestamp_utc: new Date().toISOString(),
    site: { name: "Daedalus Crater Far Side", lat: -5.9, lon: 179.4, dem: "Cesium Moon + NASA LRO SLDEM2015" },
    training_convergence: {
      algorithm: "CTDE MAPPO + 5-MLP World Model Ensemble",
      episodes: 500000,
      total_steps: 18000000,
      converged_reward: 539.88,
      world_model_mse: 0.0137,
      monte_carlo_verifier_pass_rate: 1.0,
    },
    hoppers: [1, 2, 3, 4].map((id) => ({
      hopper_id: id,
      waypoints: ["c1_start", "c2_alpha", "c3_beta", "c4_end"],
      fuel_remaining_pct: 38.0 + id * 1.5,
      min_battery_pct: 22.0 + id,
      argon_r1_r5_status: "PASS",
    })),
  };

  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "SELENE_HOPPER_UPLINK_PACKAGE.json";
  a.click();
  URL.revokeObjectURL(url);

  const statusEl = document.getElementById("export-status-text");
  if (statusEl) {
    statusEl.textContent = "UPLINK PACKAGE EXPORTED TO LOCAL HOPPER STORAGE ✓";
  }
  showToast("EXPORTED SELENE_HOPPER_UPLINK_PACKAGE.JSON FOR LOCAL DRONE UPLINK");
});

document.getElementById("btn-close-summary").addEventListener("click", () => {
  setStage("SIM_PATHS");
  scene.tiltToSimulationView();
});

document.getElementById("btn-restart-demo").addEventListener("click", () => {
  setStage("HOME");
  showToast("RESET TO STAGE 1 HOME GLOBE");
});

// ============================================================================
// MINI MOON / ORIENTATION INDICATOR + LIVE HUD TELEMETRY LOOP
// ============================================================================
const miniCanvas = document.getElementById("mini-moon-canvas");
const miniCtx = miniCanvas ? miniCanvas.getContext("2d") : null;
const miniModeEl = document.getElementById("mini-moon-mode");
const miniAnglesEl = document.getElementById("mini-moon-angles");
const miniAltEl = document.getElementById("mini-moon-alt");

// HUD Telemetry Elements
const hudHopperName = document.getElementById("flight-hopper-name");
const hudHopPhase = document.getElementById("flight-hop-phase");
const hudFuelVal = document.getElementById("hud-fuel-val");
const hudFuelBar = document.getElementById("hud-fuel-bar");
const hudBattVal = document.getElementById("hud-batt-val");
const hudBattBar = document.getElementById("hud-batt-bar");
const hudAltVal = document.getElementById("hud-alt-val");
const hudSpeedVal = document.getElementById("hud-speed-val");
const hudAttVal = document.getElementById("hud-att-val");
const hudThrustVal = document.getElementById("hud-thrust-val");
const hudTempVal = document.getElementById("hud-temp-val");
const hudDriftVal = document.getElementById("hud-drift-val");

function drawMiniMoonIndicator(telemetry) {
  if (!miniCtx || !miniCanvas) return;
  const w = miniCanvas.width;
  const h = miniCanvas.height;
  const cx = w / 2;
  const cy = h / 2;
  const r = 30;

  miniCtx.clearRect(0, 0, w, h);

  // Outer moon sphere ring
  miniCtx.strokeStyle = "rgba(255,255,255,0.55)";
  miniCtx.lineWidth = 1.4;
  miniCtx.beginPath();
  miniCtx.arc(cx, cy, r, 0, Math.PI * 2);
  miniCtx.stroke();

  // Latitude / longitude ellipses rotated by camera azimuth/pitch
  const azRad = ((telemetry.azimuthDeg || 0) * Math.PI) / 180;
  const pitchRatio = Math.max(0.15, Math.min(0.95, Math.abs(telemetry.pitchDeg || 45) / 90));

  miniCtx.strokeStyle = "rgba(255,255,255,0.22)";
  miniCtx.lineWidth = 1;
  miniCtx.beginPath();
  miniCtx.ellipse(cx, cy, r, r * pitchRatio, 0, 0, Math.PI * 2);
  miniCtx.stroke();

  miniCtx.beginPath();
  miniCtx.ellipse(cx, cy, r * 0.45, r, azRad * 0.4, 0, Math.PI * 2);
  miniCtx.stroke();

  // Camera look-vector dot on lunar disc
  const vx = cx + Math.sin(azRad) * (r * 0.55) * (1 - pitchRatio * 0.5);
  const vy = cy - Math.cos(azRad) * (r * 0.55) * (1 - pitchRatio * 0.5);

  miniCtx.fillStyle = "#3ddc84";
  miniCtx.beginPath();
  miniCtx.arc(vx, vy, 3.5, 0, Math.PI * 2);
  miniCtx.fill();

  // Crosshair ticks
  miniCtx.strokeStyle = "rgba(255,255,255,0.4)";
  miniCtx.beginPath();
  miniCtx.moveTo(cx - 5, cy);
  miniCtx.lineTo(cx + 5, cy);
  miniCtx.moveTo(cx, cy - 5);
  miniCtx.lineTo(cx, cy + 5);
  miniCtx.stroke();
}

function updateHudLoop() {
  const t = scene.getCameraTelemetry();

  if (currentStage !== "HOME" && currentStage !== "SUMMARY") {
    drawMiniMoonIndicator(t);
    miniModeEl.textContent =
      t.pitchDeg < -72 ? "TOP · DAEDALUS" : scene.manualMode ? "6-DOF MANUAL" : "3D · DAEDALUS";
    miniAnglesEl.textContent = `PITCH ${t.pitchDeg}° · AZM ${String((t.azimuthDeg + 360) % 360).padStart(3, "0")}°`;
    miniAltEl.textContent = scene.manualMode
      ? `ALT ${t.altitudeKm} KM · FREE 6-DOF`
      : `ALT ${t.altitudeKm} KM · HORIZON LOCKED`;
  }

  // Update Flight / Manual HUD card if visible
  if (currentStage === "FLIGHT") {
    if (scene.manualMode) {
      const m = scene.manualState;
      const pitchDeg = Math.round((m.euler.x * 180) / Math.PI);
      const rollDeg = Math.round((m.euler.z * 180) / Math.PI);
      const speed = m.vel.length().toFixed(1);
      const altMeters = Math.max(0, (m.pos.y - scene.sampleTerrainHeight(m.pos.x, m.pos.z)) * 2.2).toFixed(1);

      hudHopperName.textContent = `HOPPER ${scene.selectedHopperId || 1} · 6-DOF MANUAL`;
      hudHopPhase.textContent = m.crashed ? "CRASHED (PRESS R)" : "FREE ATTITUDE";
      hudHopPhase.style.color = m.crashed ? "var(--red)" : "var(--amber)";

      hudFuelVal.textContent = `${m.fuel.toFixed(1)}%`;
      hudFuelBar.style.width = `${m.fuel}%`;
      hudBattVal.textContent = `${m.battery.toFixed(1)}%`;
      hudBattBar.style.width = `${m.battery}%`;
      hudAltVal.textContent = `${altMeters} m`;
      hudSpeedVal.textContent = `${speed} m/s`;
      hudAttVal.textContent = `${pitchDeg}° / ${rollDeg}°`;
      hudThrustVal.textContent = scene.manualKeys.space
        ? "[0.00, 1.00, 0.00] FULL"
        : scene.manualKeys.shift
        ? "[0.00, -0.65, 0.00] RETRO"
        : "[0.00, 0.00, 0.00] CUT";
      hudTempVal.textContent = `${m.tempK.toFixed(0)} K`;
      hudDriftVal.textContent = m.crashed ? "OFF-NOMINAL (CRASH)" : "MANUAL OVERRIDE";
    } else {
      const prog = scene.flightProgress || 0;
      const hopIdx = Math.min(3, Math.floor(prog * 3) + 1);
      const fuel = Math.max(38, 92 - prog * 54);
      const batt = Math.max(22, 94 - prog * 28);
      const alt = Math.abs(Math.sin(prog * Math.PI * 3)) * 165 + 4.2;
      const spd = (14 + Math.sin(prog * Math.PI * 6) * 8).toFixed(1);
      const pitch = (Math.cos(prog * Math.PI * 3) * 8).toFixed(1);

      hudHopperName.textContent = `HOPPER ${scene.selectedHopperId || 1} · AUTONOMOUS`;
      hudHopPhase.textContent = prog >= 0.99 ? "TOUCHDOWN c4 ✓" : `HOP ${hopIdx}/3 → c${hopIdx + 1}`;
      hudHopPhase.style.color = "var(--green)";

      hudFuelVal.textContent = `${fuel.toFixed(1)}%`;
      hudFuelBar.style.width = `${fuel}%`;
      hudBattVal.textContent = `${batt.toFixed(1)}%`;
      hudBattBar.style.width = `${batt}%`;
      hudAltVal.textContent = `${alt.toFixed(1)} m`;
      hudSpeedVal.textContent = `${spd} m/s`;
      hudAttVal.textContent = `${pitch}° / 0.2°`;
      hudThrustVal.textContent = `[0.04, ${(0.62 + Math.sin(prog * 18) * 0.25).toFixed(2)}, -0.01]`;
      hudTempVal.textContent = `${Math.round(195 + prog * 48)} K`;
      hudDriftVal.textContent = `${(0.6 + Math.sin(prog * 9) * 0.4).toFixed(1)} m (NOMINAL)`;
    }
  }

  requestAnimationFrame(updateHudLoop);
}

// Start at Stage 1 Home Globe
setStage("HOME");
requestAnimationFrame(updateHudLoop);
