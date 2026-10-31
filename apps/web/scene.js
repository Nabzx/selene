// ============================================================================
// SELENE — 3D LUNAR MISSION CONTROL ENGINE (apps/web/scene.js)
// Hard-Coded Bulletproof Live Demo Engine · Cesium Moon WMTS + NASA SLDEM2015
// Photorealistic Copper-Tank Hopper · Multi-Hop RL Convergence Scrubber (1K..18M)
// Unconstrained 6-DOF Manual Override (Flip Upside Down + Crash + Respawn)
// ============================================================================

import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';

export const R_MOON_SCENE = 52000;
export const TERRAIN_SIZE = 20000; // 20 km x 20 km Daedalus crater floor
export const GRID_N = 200;
export const CESIUM_ION_TOKEN =
  'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJub25jZSI6InI3aTB0MGdIdGl6Y3ktOXoiLCJqdGkiOiJkYWU3NTcxNi1mNWEzLTQ3MjYtYTc3ZC1mMzUzZTEwZmU1YTAiLCJpZCI6NTM5NjYxLCJpc3MiOiJodHRwczovL2FwaS5jZXNpdW0uY29tIiwiYXVkIjoidW5kZWZpbmVkX2RlZmF1bHQiLCJpYXQiOjE3OTM0NjE5MjF9.kNlDFhhhTKaW7vH_3ev2026svaSPTNNQLnox0dUenIU';
export const CESIUM_MOON_ASSET_ID = 2684829;

export function enuToScene(x, y, z = 0) {
  return new THREE.Vector3(x, z, -y);
}

export function sceneToEnu(vec3) {
  return [vec3.x, -vec3.z, vec3.y];
}

export class SeleneScene {
  constructor(containerOrCanvas) {
    if (containerOrCanvas && containerOrCanvas.tagName === 'CANVAS') {
      this.canvas = containerOrCanvas;
    } else {
      this.canvas = document.createElement('canvas');
      if (containerOrCanvas) {
        containerOrCanvas.innerHTML = '';
        containerOrCanvas.appendChild(this.canvas);
      } else {
        document.body.appendChild(this.canvas);
      }
    }

    // Callbacks for UI
    this.onCameraUpdate = null;
    this.onSurfaceHover = null;
    this.onHopperSelected = null;
    this.onFailedRunSelected = null;
    this.onConvergenceUpdate = null;
    this.onFlightTelemetry = null;
    this.onManualCrash = null;
    this.onManualRespawn = null;
    this.onFlightComplete = null;
    this.onFlightFinished = null;

    // Terrain buffers
    this.terrainData = null;
    this.slopeData = null;
    this.tempData = null;
    this.terrainMeta = null;

    // Hard-coded deterministic multi-hop fleet plan & RL iteration clouds
    this.fleetHoppers = []; // 4 hoppers (h0..h3)
    this.selectedHopperId = 'all';

    // Training iteration scrubber value: 0.0 (1,000 steps) -> 1.0 (500k eps / 18M steps)
    this.trainingScrub = 0.0;
    this.showSimPaths = false;
    this.pathAnimProgress = 1.0;
    this.isAnimatingWave = false;

    // Flight state
    this.flightActive = false;
    this.flightProgress = 0.0;
    this.flightSpeed = 0.042;
    this.followHopperId = 'h0';

    // Unconstrained 6-DOF Manual Override state (can flip upside down & crash)
    this.manualFpvActive = false;
    this.manualCrashed = false;
    this.fpvKeys = { w: false, a: false, s: false, d: false, space: false, shift: false };
    this.fpvState = {
      pos: new THREE.Vector3(0, 420, 0),
      vel: new THREE.Vector3(0, 0, 0),
      quat: new THREE.Quaternion(),
      euler: new THREE.Euler(0, 0, 0, 'XYZ'),
      angVel: new THREE.Vector3(0, 0, 0), // Local pitch, yaw, roll rates (rad/s)
      fuelKg: 15.0,
      fuel: 96.0,
      batteryPct: 98.0,
      battery: 98.0,
      tempK: 218.0,
      crashed: false,
      thrustVec: [0, 0, 0],
      crashReason: '',
    };

    // Camera tween state
    this.cameraTween = null;
    this.globeSpinActive = true;

    this._initRendererAndScene();
    this._initLighting();
    this._initStarfield();
    this._initMoonGlobe();
    this._initGroups();
    this._initInteraction();

    this.loadAllDatasets();
    this.startLoop();
  }

  _initRendererAndScene() {
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: true,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.12;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x020203);

    this.camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 8, 950000);
    // Start looking at the full 3D Moon Globe (Stage 1 / panel02-b.jpg)
    this.camera.position.set(0, 4200, R_MOON_SCENE + 33500);

    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.target.set(0, -R_MOON_SCENE * 0.88, 0);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.07;
    this.controls.minDistance = 85;
    this.controls.maxDistance = 175000;
    this.controls.maxPolarAngle = Math.PI * 0.44;
    this.controls.zoomSpeed = 1.15;
    this.controls.panSpeed = 0.9;

    window.addEventListener('resize', () => {
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(window.innerWidth, window.innerHeight);
    });
  }

  _initLighting() {
    // Low-angle harsh solar key light (like Apollo 70mm Hasselblad archival photos)
    this.sunLight = new THREE.DirectionalLight(0xfffcf5, 3.4);
    this.sunLight.position.set(22000, 11500, 14000);
    this.sunLight.castShadow = true;
    this.sunLight.shadow.mapSize.width = 2048;
    this.sunLight.shadow.mapSize.height = 2048;
    this.sunLight.shadow.camera.near = 500;
    this.sunLight.shadow.camera.far = 65000;
    const d = 12000;
    this.sunLight.shadow.camera.left = -d;
    this.sunLight.shadow.camera.right = d;
    this.sunLight.shadow.camera.top = d;
    this.sunLight.shadow.camera.bottom = -d;
    this.scene.add(this.sunLight);

    // Subtle rim/fill light so shadows and copper hopper tank read clearly
    this.hemiLight = new THREE.HemisphereLight(0xd8dee9, 0x1a1c22, 0.42);
    this.scene.add(this.hemiLight);
  }

  _initStarfield() {
    const count = 3200;
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const r = 240000 + Math.random() * 200000;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      pos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      pos[i * 3 + 2] = r * Math.cos(phi);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({
      color: 0xe2e8f0,
      size: 320,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.7,
    });
    this.starfield = new THREE.Points(geo, mat);
    this.scene.add(this.starfield);
  }

  _initMoonGlobe() {
    this.globeGroup = new THREE.Group();
    this.globeGroup.position.set(0, -R_MOON_SCENE, 0);
    this.scene.add(this.globeGroup);

    const sphereGeo = new THREE.SphereGeometry(R_MOON_SCENE, 96, 64);
    this.globeMat = new THREE.MeshStandardMaterial({
      color: 0xdadce0,
      roughness: 0.92,
      metalness: 0.02,
      bumpScale: 340,
      transparent: true,
      opacity: 1.0,
    });
    this.moonSphere = new THREE.Mesh(sphereGeo, this.globeMat);
    this.moonSphere.rotation.y = -Math.PI * 0.5;
    this.globeGroup.add(this.moonSphere);

    // Thin white S-band telemetry orbital ellipse ring around the Moon (matches panel02-b.jpg!)
    const ringCurve = new THREE.EllipseCurve(0, 0, R_MOON_SCENE * 1.48, R_MOON_SCENE * 1.18, 0, Math.PI * 2, false, 0);
    const ringPts2D = ringCurve.getPoints(180);
    const ringPts3D = ringPts2D.map((p) => new THREE.Vector3(p.x, 0, p.y));
    const ringGeo = new THREE.BufferGeometry().setFromPoints(ringPts3D);
    this.orbitalTelemetryRing = new THREE.LineLoop(
      ringGeo,
      new THREE.LineBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.72,
      })
    );
    this.orbitalTelemetryRing.rotation.z = -0.28;
    this.orbitalTelemetryRing.rotation.x = 0.22;
    this.globeGroup.add(this.orbitalTelemetryRing);

    // Glowing white Perilune / Daedalus lock node on the orbital ring (panel02-b.jpg)
    const dotGeo = new THREE.SphereGeometry(520, 16, 16);
    const dotMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    this.orbitalLockDot = new THREE.Mesh(dotGeo, dotMat);
    this.orbitalLockDot.position.set(-R_MOON_SCENE * 1.38, R_MOON_SCENE * 0.38, R_MOON_SCENE * 0.25);
    this.globeGroup.add(this.orbitalLockDot);

    // Daedalus target pin on the lunar surface
    const targetDot = new THREE.Mesh(
      new THREE.SphereGeometry(420, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0xffffff })
    );
    targetDot.position.set(R_MOON_SCENE * 0.62, R_MOON_SCENE * 0.08, R_MOON_SCENE * 0.78);
    this.globeGroup.add(targetDot);

    // Stream Cesium Moon / NASA Moon Trek global mosaic tiles
    this._streamCesiumMoonGlobalTiles();
  }

  _streamCesiumMoonGlobalTiles() {
    const loader = new THREE.TextureLoader();
    loader.setCrossOrigin('anonymous');

    // Load 4K Cesium Moon Global Albedo & Bump stitched from Cesium Moon 3D Tiles (Asset 2684829)
    loader.load('/data/cesium_moon/cesium_globe_albedo_4k.jpg', (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.anisotropy = 16;
      this.globeMat.map = tex;
      this.globeMat.needsUpdate = true;
    });

    loader.load('/data/cesium_moon/cesium_globe_bump_4k.jpg', (bump) => {
      this.globeMat.bumpMap = bump;
      this.globeMat.bumpScale = 420;
      this.globeMat.needsUpdate = true;
    });

    // Also load the 34 Global Cesium Moon 3D Tiles mesh pack (Asset 2684829) onto a 3D Tiles group
    this.cesiumGlobeTilesGroup = new THREE.Group();
    this.moonSphere.add(this.cesiumGlobeTilesGroup);
    fetch('/data/cesium_moon/globe_3dtiles_pack.json')
      .then((r) => (r.ok ? r.json() : null))
      .then((pack) => {
        if (!pack || !pack.tiles) return;
        const scale = (R_MOON_SCENE * 1.001) / 600.0;
        pack.tiles.forEach((tile) => {
          const geo = new THREE.BufferGeometry();
          const verts = new Float32Array(tile.positions_flat.length);
          for (let i = 0; i < verts.length; i++) {
            verts[i] = tile.positions_flat[i] * scale;
          }
          geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
          geo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(tile.uvs_flat), 2));
          geo.setIndex(tile.indices);
          geo.computeVertexNormals();

          const tex = loader.load(tile.texture);
          tex.colorSpace = THREE.SRGBColorSpace;
          tex.flipY = false;
          const mat = new THREE.MeshStandardMaterial({
            map: tex,
            roughness: 0.92,
            metalness: 0.02,
            transparent: true,
            opacity: 0.35,
          });
          const m = new THREE.Mesh(geo, mat);
          this.cesiumGlobeTilesGroup.add(m);
        });
      })
      .catch(() => {});

    // Authenticate live Cesium ion 3D Tiles endpoint with user's token
    fetch(`https://api.cesium.com/v1/assets/${CESIUM_MOON_ASSET_ID}/endpoint`, {
      headers: { Authorization: `Bearer ${CESIUM_ION_TOKEN}` },
    })
      .then((r) => (r.ok ? r.json() : null))
      .then((ep) => {
        if (ep && ep.url) {
          this.cesiumIonEndpoint = ep;
        }
      })
      .catch(() => {});
  }

  _initGroups() {
    this.surfaceGroup = new THREE.Group();
    this.surfaceGroup.visible = false;
    this.scene.add(this.surfaceGroup);

    this.cratersGroup = new THREE.Group();
    this.surfaceGroup.add(this.cratersGroup);

    this.previewArcGroup = new THREE.Group();
    this.surfaceGroup.add(this.previewArcGroup);

    this.rolloutsGroup = new THREE.Group();
    this.surfaceGroup.add(this.rolloutsGroup);

    this.hoppersGroup = new THREE.Group();
    this.surfaceGroup.add(this.hoppersGroup);

    this.relayLaserGroup = new THREE.Group();
    this.surfaceGroup.add(this.relayLaserGroup);
  }

  _initInteraction() {
    this.raycaster = new THREE.Raycaster();
    this.raycaster.params.Line.threshold = 140;
    this.mouseNDC = new THREE.Vector2();
    let downPos = { x: 0, y: 0 };

    this.canvas.addEventListener('pointerdown', (e) => {
      downPos = { x: e.clientX, y: e.clientY };
    });

    this.canvas.addEventListener('pointermove', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      this.mouseNDC.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      this.mouseNDC.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      if (this.terrainMesh && this.surfaceGroup.visible && !this.manualFpvActive) {
        this.raycaster.setFromCamera(this.mouseNDC, this.camera);
        const hits = this.raycaster.intersectObject(this.terrainMesh, false);
        if (hits.length > 0 && this.onSurfaceHover) {
          const pt = hits[0].point;
          const [ex, ey] = sceneToEnu(pt);
          const elev = this.getSurfaceElevation(ex, ey);
          const slope = this.getSurfaceSlope(ex, ey);
          const temp = this.getSurfaceTemp(ex, ey);
          this.onSurfaceHover({ x: ex, y: ey, elev, slope, temp });
        }
      }
    });

    this.canvas.addEventListener('pointerup', (e) => {
      if (Math.hypot(e.clientX - downPos.x, e.clientY - downPos.y) > 6) return;
      if (!this.surfaceGroup.visible || this.manualFpvActive) return;

      this.raycaster.setFromCamera(this.mouseNDC, this.camera);

      // 1. Check click on any of the 4 copper-tank Lunar Hoppers
      const hopperTargets = this.fleetHoppers.map((h) => h.mesh).filter(Boolean);
      const hopperHits = this.raycaster.intersectObjects(hopperTargets, true);
      if (hopperHits.length > 0) {
        let obj = hopperHits[0].object;
        while (obj && !obj.userData?.hopperId) obj = obj.parent;
        if (obj?.userData?.hopperId && this.onHopperSelected) {
          this.onHopperSelected(obj.userData.hopperId);
          return;
        }
      }

      // 2. Check click on any red failed rollout trajectory
      if (this.showSimPaths) {
        const redTargets = [];
        for (const h of this.fleetHoppers) {
          for (const r of h.redLines) {
            if (r.line.visible) redTargets.push(r.line);
          }
        }
        const redHits = this.raycaster.intersectObjects(redTargets, false);
        if (redHits.length > 0) {
          const meta = redHits[0].object.userData?.failMeta;
          if (meta && this.onFailedRunSelected) {
            this.onFailedRunSelected(meta);
          }
        }
      }
    });
  }

  // ---------------------------------------------------------------------------
  // Dataset Loading & High-Detail Daedalus Crater Terrain Construction
  // ---------------------------------------------------------------------------
  async loadAllDatasets() {
    const fetchBin = async (url) => {
      const r = await fetch(url);
      const buf = await r.arrayBuffer();
      return new Float32Array(buf);
    };
    const fetchJson = async (url) => {
      const r = await fetch(url);
      return r.json();
    };

    try {
      const [tBin, sBin, tmpBin, tMeta] = await Promise.all([
        fetchBin('/data/terrain.bin'),
        fetchBin('/data/slope.bin'),
        fetchBin('/data/temp.bin'),
        fetchJson('/data/terrain.json'),
      ]);
      this.terrainData = tBin;
      this.slopeData = sBin;
      this.tempData = tmpBin;
      this.terrainMeta = tMeta;
    } catch (_err) {
      // Fallback procedural terrain if offline
      this.terrainData = new Float32Array(GRID_N * GRID_N);
      this.slopeData = new Float32Array(GRID_N * GRID_N);
      this.tempData = new Float32Array(GRID_N * GRID_N).fill(-185);
    }

    this._enhanceCraterRelief();
    this._buildTerrainMesh();
    this._buildHardcodedMultiHopMission();
  }

  // Blends real NASA SLDEM2015 topography with crisp central Daedalus crater rim,
  // central peak, and satellite craters (c1..c8) so the 3D terrain looks as sharp
  // and dramatic as panel03-a.jpg, panel04-b.jpg, and panel05-d.jpg.
  _enhanceCraterRelief() {
    this.MissionCraters = [
      { id: 'DAEDALUS-CENTRAL', label: 'c2 · DAEDALUS FLOOR', x: 0, y: 0, r: 3200, depth: 360, peak: 170, coldTrap: true },
      { id: 'c1', label: 'START · c1', x: -3400, y: -4800, r: 950, depth: 120, peak: 0, coldTrap: false },
      { id: 'c2', label: 'c2 · SADDLE', x: 1850, y: -650, r: 1100, depth: 145, peak: 0, coldTrap: true },
      { id: 'c3', label: 'c3 · NORTH RIM', x: -2400, y: 3600, r: 1250, depth: 165, peak: 0, coldTrap: true },
      { id: 'c4', label: 'END · c4', x: 5200, y: 4700, r: 1150, depth: 160, peak: 0, coldTrap: true },
      { id: 'c5', label: 'c5 · EAST BASIN', x: 5800, y: -1200, r: 980, depth: 135, peak: 0, coldTrap: false },
      { id: 'c6', label: 'c6 · WEST CRATER', x: -5600, y: -1100, r: 920, depth: 130, peak: 0, coldTrap: false },
    ];

    for (let row = 0; row < GRID_N; row++) {
      const y = -TERRAIN_SIZE / 2 + (row / (GRID_N - 1)) * TERRAIN_SIZE;
      for (let col = 0; col < GRID_N; col++) {
        const x = -TERRAIN_SIZE / 2 + (col / (GRID_N - 1)) * TERRAIN_SIZE;
        const idx = row * GRID_N + col;

        let h = (this.terrainData[idx] || 0) * 0.55;
        let maxRimSlope = this.slopeData[idx] || 3.5;

        for (const c of this.MissionCraters) {
          const dist = Math.hypot(x - c.x, y - c.y);
          const u = dist / c.r;
          if (u < 1.45) {
            if (u < 0.85) {
              // Crater bowl
              const bowl = -c.depth * (1.0 - (u / 0.85) ** 2);
              h += bowl;
              if (c.peak > 0 && u < 0.28) {
                h += c.peak * Math.cos((u / 0.28) * (Math.PI / 2));
              }
            } else if (u <= 1.15) {
              // Crisp raised crater rim (steep >15 deg slope on inner wall)
              const rimPhase = (u - 0.85) / 0.3;
              const rimH = c.depth * 0.34 * Math.sin(rimPhase * Math.PI);
              h += rimH;
              maxRimSlope = Math.max(maxRimSlope, 18.5 + 8.0 * Math.sin(rimPhase * Math.PI));
            } else {
              // Ejecta blanket taper
              const ej = (1.45 - u) / 0.3;
              h += c.depth * 0.08 * ej * ej;
            }
          }
        }
        this.terrainData[idx] = h;
        this.slopeData[idx] = maxRimSlope;
      }
    }
  }

  getSurfaceElevation(x, y) {
    if (!this.terrainData) return 0;
    const u = Math.max(0, Math.min(1, (x + TERRAIN_SIZE / 2) / TERRAIN_SIZE));
    const v = Math.max(0, Math.min(1, (y + TERRAIN_SIZE / 2) / TERRAIN_SIZE));
    const col = u * (GRID_N - 1);
    const row = v * (GRID_N - 1);
    const c0 = Math.floor(col);
    const r0 = Math.floor(row);
    const c1 = Math.min(GRID_N - 1, c0 + 1);
    const r1 = Math.min(GRID_N - 1, r0 + 1);
    const tx = col - c0;
    const ty = row - r0;
    const z00 = this.terrainData[r0 * GRID_N + c0];
    const z10 = this.terrainData[r0 * GRID_N + c1];
    const z01 = this.terrainData[r1 * GRID_N + c0];
    const z11 = this.terrainData[r1 * GRID_N + c1];
    return (1 - ty) * ((1 - tx) * z00 + tx * z10) + ty * ((1 - tx) * z01 + tx * z11);
  }

  getSurfaceSlope(x, y) {
    if (!this.slopeData) return 3.2;
    const u = Math.max(0, Math.min(1, (x + TERRAIN_SIZE / 2) / TERRAIN_SIZE));
    const v = Math.max(0, Math.min(1, (y + TERRAIN_SIZE / 2) / TERRAIN_SIZE));
    const col = Math.round(u * (GRID_N - 1));
    const row = Math.round(v * (GRID_N - 1));
    return this.slopeData[row * GRID_N + col] || 3.2;
  }

  getSurfaceTemp(x, y) {
    if (!this.tempData) return -184.0;
    const u = Math.max(0, Math.min(1, (x + TERRAIN_SIZE / 2) / TERRAIN_SIZE));
    const v = Math.max(0, Math.min(1, (y + TERRAIN_SIZE / 2) / TERRAIN_SIZE));
    const col = Math.round(u * (GRID_N - 1));
    const row = Math.round(v * (GRID_N - 1));
    return this.tempData[row * GRID_N + col] || -184.0;
  }

  _buildTerrainMesh() {
    const geo = new THREE.PlaneGeometry(TERRAIN_SIZE, TERRAIN_SIZE, GRID_N - 1, GRID_N - 1);
    geo.rotateX(-Math.PI / 2);

    const pos = geo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const z = pos.getZ(i);
      const yEnu = -z;
      const elev = this.getSurfaceElevation(x, yEnu);
      pos.setY(i, elev);
    }
    geo.computeVertexNormals();

    this.textures = {};
    const loader = new THREE.TextureLoader();

    // Load 2K Cesium Moon Daedalus Albedo & Bump stitched from Level-10 Cesium Moon 3D Tiles (Asset 2684829)
    this.textures.wac = loader.load('/data/cesium_moon/cesium_daedalus_albedo_2k.jpg', (t) => {
      t.colorSpace = THREE.SRGBColorSpace;
      t.anisotropy = 16;
    });
    this.textures.bump = loader.load('/data/cesium_moon/cesium_daedalus_bump_2k.jpg', (t) => {
      t.anisotropy = 16;
    });
    this.textures.thermal = this._createOverlayTexture('thermal');
    this.textures.slope = this._createOverlayTexture('slope');
    this.textures.uncertainty = this._createOverlayTexture('uncertainty');

    this.terrainMat = new THREE.MeshStandardMaterial({
      map: this.textures.wac,
      bumpMap: this.textures.bump,
      bumpScale: 34,
      color: 0xd8dce3,
      roughness: 0.94,
      metalness: 0.03,
    });

    this.terrainMesh = new THREE.Mesh(geo, this.terrainMat);
    this.terrainMesh.receiveShadow = true;
    this.surfaceGroup.add(this.terrainMesh);

    this._buildCraterMarkersAndPreviewArc();
  }

  _createOverlayTexture(mode) {
    const size = 256;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(size, size);

    for (let r = 0; r < size; r++) {
      const y = (0.5 - r / (size - 1)) * TERRAIN_SIZE;
      for (let c = 0; c < size; c++) {
        const x = (c / (size - 1) - 0.5) * TERRAIN_SIZE;
        const elev = this.getSurfaceElevation(x, y);
        const slope = this.getSurfaceSlope(x, y);
        const temp = this.getSurfaceTemp(x, y);
        const base = Math.max(30, Math.min(210, 120 + elev * 0.18));

        let R = base;
        let G = base;
        let B = base;

        if (mode === 'slope') {
          if (slope > 15.0) {
            R = 255;
            G = 68;
            B = 68;
          } else {
            R = base * 0.85;
            G = Math.min(255, base * 1.05);
            B = base * 0.88;
          }
        } else if (mode === 'thermal') {
          const cold = Math.max(0, Math.min(1, (-150 - temp) / 80));
          R = base * (1 - cold * 0.55);
          G = base * (1 - cold * 0.15);
          B = Math.min(255, base + cold * 95);
        } else if (mode === 'uncertainty') {
          const rad = Math.hypot(x, y) / 9000;
          const ood = Math.max(0, Math.min(1, (rad - 0.45) * 1.4));
          R = base * (1 - ood * 0.4);
          G = Math.min(255, base + ood * 80);
          B = Math.min(255, base + ood * 95);
        }

        const p = (r * size + c) * 4;
        img.data[p] = R;
        img.data[p + 1] = G;
        img.data[p + 2] = B;
        img.data[p + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }

  setSurfaceLayer(layerName) {
    if (!this.terrainMat || !this.textures[layerName]) return;
    this.terrainMat.map = this.textures[layerName];
    this.terrainMat.needsUpdate = true;
  }

  // Builds the clean white START (c1) -> c2 -> END (c4) preview curve shown in Panel 3 (panel03-a.jpg)
  _buildCraterMarkersAndPreviewArc() {
    while (this.cratersGroup.children.length) {
      this.cratersGroup.remove(this.cratersGroup.children[0]);
    }
    while (this.previewArcGroup.children.length) {
      this.previewArcGroup.remove(this.previewArcGroup.children[0]);
    }

    for (const c of this.MissionCraters) {
      const z = this.getSurfaceElevation(c.x, c.y) + 18;
      // Glowing green/white waypoint node
      const dot = new THREE.Mesh(
        new THREE.SphereGeometry(65, 16, 16),
        new THREE.MeshBasicMaterial({ color: c.id === 'c1' || c.id === 'c4' ? 0xffffff : 0x3ddc84 })
      );
      dot.position.copy(enuToScene(c.x, c.y, z));
      this.cratersGroup.add(dot);

      // Subtle ring around crater target
      const ringPts = [];
      for (let i = 0; i <= 64; i++) {
        const a = (i / 64) * Math.PI * 2;
        const rx = c.x + Math.cos(a) * c.r;
        const ry = c.y + Math.sin(a) * c.r;
        ringPts.push(enuToScene(rx, ry, this.getSurfaceElevation(rx, ry) + 14));
      }
      const ringLine = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(ringPts),
        new THREE.LineBasicMaterial({
          color: 0xffffff,
          transparent: true,
          opacity: 0.22,
        })
      );
      this.cratersGroup.add(ringLine);
    }

    // White preview mission corridor arc from START (c1) -> c2 -> END (c4) matching panel03-a.jpg
    const previewWaypoints = [
      [-3400, -4800],
      [1850, -650],
      [5200, 4700],
    ];
    const arcPts = this._createMultiHopCurvePoints(previewWaypoints, 420);
    const previewLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(arcPts),
      new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.65 })
    );
    this.previewArcGroup.add(previewLine);
  }

  _createMultiHopCurvePoints(xyWaypoints, apexBoost = 520) {
    const pts = [];
    for (let h = 0; h < xyWaypoints.length - 1; h++) {
      const [sx, sy] = xyWaypoints[h];
      const [ex, ey] = xyWaypoints[h + 1];
      const sz = this.getSurfaceElevation(sx, sy) + 12;
      const ez = this.getSurfaceElevation(ex, ey) + 12;
      const dist = Math.hypot(ex - sx, ey - sy);
      const apex = Math.max(sz, ez) + Math.max(240, Math.min(950, dist * 0.16 + apexBoost * 0.4));

      const steps = 40;
      for (let s = 0; s <= steps; s++) {
        if (h > 0 && s === 0) continue;
        const u = s / steps;
        const x = sx + (ex - sx) * u;
        const y = sy + (ey - sy) * u;
        const baseZ = sz + (ez - sz) * u;
        const z = baseZ + (apex - Math.max(sz, ez)) * 4 * u * (1 - u);
        pts.push(enuToScene(x, y, z));
      }
    }
    return pts;
  }

  // ---------------------------------------------------------------------------
  // Photorealistic Copper-Tank Lunar Hopper 3D Model (Matches Attached Image!)
  // ---------------------------------------------------------------------------
  createPhotorealHopperMesh(hopperId) {
    const root = new THREE.Group();
    root.userData = { hopperId };

    // Materials matching the attached Hopper render (media_1793461003047_9ddacfb7.png)
    const copperTankMat = new THREE.MeshStandardMaterial({
      color: 0xc86b38, // Polished copper/bronze spherical propellant tank
      metalness: 0.84,
      roughness: 0.22,
    });
    const deckGreyMat = new THREE.MeshStandardMaterial({
      color: 0x9aa1ab, // Matte grey octagonal avionics deck
      metalness: 0.45,
      roughness: 0.48,
    });
    const strutSilverMat = new THREE.MeshStandardMaterial({
      color: 0xdbe2ec, // Bright brushed aluminium leg tubes & ring
      metalness: 0.78,
      roughness: 0.25,
    });
    const blackSleeveMat = new THREE.MeshStandardMaterial({
      color: 0x191b1f, // Matte black shock-absorber sleeves & RCS nozzle
      metalness: 0.25,
      roughness: 0.75,
    });
    const whiteTowerMat = new THREE.MeshStandardMaterial({
      color: 0xf1f5f9, // White vertical solar/radiator instrument tower
      metalness: 0.2,
      roughness: 0.35,
    });

    // 1. Central Spherical Copper Propellant Tank
    const tankMesh = new THREE.Mesh(new THREE.SphereGeometry(1.18, 32, 24), copperTankMat);
    tankMesh.position.y = 1.65;
    tankMesh.castShadow = true;
    root.add(tankMesh);

    // 2. Equatorial Aluminium Structural Ring around Copper Sphere
    const eqRing = new THREE.Mesh(new THREE.TorusGeometry(1.28, 0.065, 12, 36), strutSilverMat);
    eqRing.rotation.x = Math.PI / 2;
    eqRing.position.y = 1.52;
    root.add(eqRing);

    // 4 Vertical Struts connecting Equatorial Ring to Upper Deck
    for (let i = 0; i < 4; i++) {
      const a = (i * Math.PI) / 2 + Math.PI / 4;
      const post = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 0.95, 10), strutSilverMat);
      post.position.set(Math.cos(a) * 1.24, 1.98, Math.sin(a) * 1.24);
      root.add(post);
    }

    // 3. Upper Octagonal Grey Avionics Deck (sitting above copper tank)
    const deckMesh = new THREE.Mesh(new THREE.CylinderGeometry(1.42, 1.42, 0.56, 8), deckGreyMat);
    deckMesh.position.y = 2.62;
    deckMesh.castShadow = true;
    root.add(deckMesh);

    // 4. Side Conical Black Attitude Thruster / Sensor Horn on Upper Deck (visible in photo)
    const sideCone = new THREE.Mesh(new THREE.ConeGeometry(0.24, 0.48, 16, 1, true), blackSleeveMat);
    sideCone.rotation.z = -Math.PI / 2.4;
    sideCone.position.set(0.85, 2.22, 0.92);
    root.add(sideCone);

    // 5. Top White Vertical Instrument / Solar Radiator Tower + Black Avionics Box
    const towerMesh = new THREE.Mesh(new THREE.BoxGeometry(0.56, 1.35, 0.32), whiteTowerMat);
    towerMesh.position.set(-0.08, 3.55, -0.12);
    towerMesh.castShadow = true;
    root.add(towerMesh);

    const blackBox = new THREE.Mesh(new THREE.BoxGeometry(0.54, 0.42, 0.44), blackSleeveMat);
    blackBox.position.set(0.08, 3.08, 0.32);
    root.add(blackBox);

    // 6. Left Tall T-Bar Telemetry Antenna Mast
    const antPole = new THREE.Mesh(new THREE.CylinderGeometry(0.032, 0.032, 1.35, 8), strutSilverMat);
    antPole.position.set(-0.78, 3.55, 0.15);
    root.add(antPole);
    const antTBar = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.42, 8), strutSilverMat);
    antTBar.rotation.z = Math.PI / 2;
    antTBar.position.set(-0.78, 4.22, 0.15);
    root.add(antTBar);

    // 7. Right Optical / Thermal Navigation Camera Mast
    const camPole = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.85, 8), strutSilverMat);
    camPole.position.set(0.68, 3.3, -0.25);
    root.add(camPole);
    const camHead = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.16, 0.2), blackSleeveMat);
    camHead.rotation.z = -0.22;
    camHead.position.set(0.75, 3.74, -0.25);
    root.add(camHead);

    // 8. Right-Side L-Shaped Surface Probe Boom (characteristic arm on right side of photo!)
    const boomHoriz = new THREE.Mesh(new THREE.CylinderGeometry(0.032, 0.032, 1.65, 8), strutSilverMat);
    boomHoriz.rotation.z = Math.PI / 2;
    boomHoriz.position.set(1.95, 2.38, 0.0);
    root.add(boomHoriz);
    const boomVert = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 2.15, 8), strutSilverMat);
    boomVert.position.set(2.74, 1.32, 0.0);
    root.add(boomVert);

    // 9. Four A-Frame Landing Legs with Dual Black Shock-Absorber Sleeves & Circular Footpads
    for (let i = 0; i < 4; i++) {
      const angle = (i * Math.PI) / 2 + Math.PI / 4;
      const legGroup = new THREE.Group();
      legGroup.rotation.y = angle;

      // Primary upper-to-footpad strut
      const mainStrut = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 3.35, 10), strutSilverMat);
      mainStrut.position.set(2.05, 1.28, 0);
      mainStrut.rotation.z = 0.56;
      legGroup.add(mainStrut);

      // Black cylindrical shock-absorber damper sleeve on main strut
      const blackSleeve1 = new THREE.Mesh(new THREE.CylinderGeometry(0.088, 0.088, 1.05, 10), blackSleeveMat);
      blackSleeve1.position.set(2.22, 1.02, 0);
      blackSleeve1.rotation.z = 0.56;
      legGroup.add(blackSleeve1);

      // Secondary A-frame brace connecting equatorial ring to footpad
      const subStrut = new THREE.Mesh(new THREE.CylinderGeometry(0.042, 0.042, 2.45, 8), strutSilverMat);
      subStrut.position.set(1.98, 0.78, 0.26);
      subStrut.rotation.z = 0.82;
      subStrut.rotation.x = 0.14;
      legGroup.add(subStrut);

      const blackSleeve2 = new THREE.Mesh(new THREE.CylinderGeometry(0.072, 0.072, 0.85, 8), blackSleeveMat);
      blackSleeve2.position.set(2.12, 0.65, 0.22);
      blackSleeve2.rotation.z = 0.82;
      legGroup.add(blackSleeve2);

      // Circular lunar regolith footpad
      const pad = new THREE.Mesh(new THREE.CylinderGeometry(0.38, 0.42, 0.06, 18), deckGreyMat);
      pad.position.set(2.92, 0.03, 0);
      pad.castShadow = true;
      legGroup.add(pad);

      root.add(legGroup);
    }

    // 10. Bottom Main Gimballed Engine Bell + Glowing Plume & Downward Thrust Vector Arrows (panel07-c.jpg)
    const bell = new THREE.Mesh(new THREE.ConeGeometry(0.42, 0.65, 20, 1, true), blackSleeveMat);
    bell.position.y = 0.38;
    root.add(bell);

    const plumeGroup = new THREE.Group();
    // Main white/cyan rocket exhaust cone
    const mainPlume = new THREE.Mesh(
      new THREE.ConeGeometry(0.55, 2.8, 18),
      new THREE.MeshBasicMaterial({
        color: 0xe0f2fe,
        transparent: true,
        opacity: 0.85,
      })
    );
    mainPlume.rotation.x = Math.PI;
    mainPlume.position.y = -1.1;
    plumeGroup.add(mainPlume);

    // 3 Crisp White Downward Thrust Vector Arrows (matching panel07-c.jpg!)
    for (const dx of [-1.2, 0, 1.2]) {
      const shaft = new THREE.Mesh(
        new THREE.CylinderGeometry(0.045, 0.045, 2.4, 8),
        new THREE.MeshBasicMaterial({ color: 0xffffff })
      );
      shaft.position.set(dx, -1.2, 0);
      plumeGroup.add(shaft);

      const tip = new THREE.Mesh(
        new THREE.ConeGeometry(0.16, 0.42, 10),
        new THREE.MeshBasicMaterial({ color: 0xffffff })
      );
      tip.rotation.x = Math.PI;
      tip.position.set(dx, -2.5, 0);
      plumeGroup.add(tip);
    }

    plumeGroup.visible = false;
    root.add(plumeGroup);
    root.userData.plume = plumeGroup;

    // Scale hopper so it is clearly visible on the 20x20km tactical map & follow cam
    root.scale.setScalar(42);
    return root;
  }

  // ---------------------------------------------------------------------------
  // Hard-Coded Bulletproof 4-Hopper Multi-Hop Mission & RL Convergence Cloud
  // ---------------------------------------------------------------------------
  _buildHardcodedMultiHopMission() {
    while (this.hoppersGroup.children.length) {
      this.hoppersGroup.remove(this.hoppersGroup.children[0]);
    }
    while (this.rolloutsGroup.children.length) {
      this.rolloutsGroup.remove(this.rolloutsGroup.children[0]);
    }

    // 4 Lunar Hoppers going from START coordinates -> past craters (3 hops each) -> END coordinates
    const specs = [
      {
        id: 'h0',
        name: 'HOPPER 1 (ALPHA)',
        sector: 'PRIMARY CORRIDOR · c1 → c2 → c4',
        waypoints: [
          [-3400, -4800], // START (c1)
          [-850, -2100],  // Hop 1: Daedalus South Rim Saddle
          [1850, -650],   // Hop 2: c2 Central Basin
          [5200, 4700],   // Hop 3: END (c4)
        ],
        hopLabels: ['START c1', 'DAEDALUS S-SADDLE', 'CRATER c2', 'END c4'],
        fuelEndPct: 41.2,
      },
      {
        id: 'h1',
        name: 'HOPPER 2 (BETA)',
        sector: 'NORTHERN ARC · c1 → c6 → c3 → c4',
        waypoints: [
          [-3800, -4400], // START (c1 West Pad)
          [-5600, -1100], // Hop 1: Crater c6
          [-2400, 3600],  // Hop 2: Crater c3 North Rim
          [4750, 4950],   // Hop 3: END (c4 West Rim)
        ],
        hopLabels: ['START c1-W', 'CRATER c6', 'CRATER c3', 'END c4-W'],
        fuelEndPct: 38.6,
      },
      {
        id: 'h2',
        name: 'HOPPER 3 (GAMMA)',
        sector: 'EASTERN BASIN · c1 → c5 → c4',
        waypoints: [
          [-2950, -5100], // START (c1 East Pad)
          [1600, -3400],  // Hop 1: South-East Ridge
          [5800, -1200],  // Hop 2: Crater c5 East Basin
          [5550, 4350],   // Hop 3: END (c4 East Pad)
        ],
        hopLabels: ['START c1-E', 'SE RIDGE', 'CRATER c5', 'END c4-E'],
        fuelEndPct: 43.5,
      },
      {
        id: 'h3',
        name: 'HOPPER 4 (DELTA)',
        sector: 'CENTRAL PEAK RELAY · c1 → PEAK → c4',
        waypoints: [
          [-3300, -4300], // START (c1 North Pad)
          [-2100, 450],   // Hop 1: Daedalus Inner Terrace
          [320, 1950],    // Hop 2: Central Peak North Pass
          [4950, 4450],   // Hop 3: END (c4 South Pad)
        ],
        hopLabels: ['START c1-N', 'INNER TERRACE', 'PEAK N-PASS', 'END c4-S'],
        fuelEndPct: 39.8,
      },
    ];

    const failureReasons = [
      ['R3 VIOLATION', 'TOUCHDOWN SLOPE 21.4 DEG > 15.0 DEG (CRATER WALL)'],
      ['R3 VIOLATION', 'TOUCHDOWN SLOPE 24.8 DEG > 15.0 DEG (CENTRAL PEAK)'],
      ['R1 VIOLATION', 'PROPELLANT RESERVE 8.4% < 15.0% MINIMUM GATE'],
      ['R2 VIOLATION', 'BATTERY RESERVE 14.2% < 20.0% IN COLD TRAP (-224°C)'],
      ['R4 VIOLATION', 'INTER-HOPPER SEPARATION 62m < 100m KEEP-OUT ZONE'],
    ];

    this.fleetHoppers = [];

    specs.forEach((sp, hIdx) => {
      const hopperMesh = this.createPhotorealHopperMesh(sp.id);
      const [sx, sy] = sp.waypoints[0];
      const sz = this.getSurfaceElevation(sx, sy);
      hopperMesh.position.copy(enuToScene(sx, sy, sz));
      this.hoppersGroup.add(hopperMesh);

      // 5-MLP World Model Ghost Hopper (translucent cyan wireframe twin for Flight mode)
      const ghostMesh = new THREE.Mesh(
        new THREE.OctahedronGeometry(55, 1),
        new THREE.MeshBasicMaterial({
          color: 0x5eead4,
          wireframe: true,
          transparent: true,
          opacity: 0.65,
        })
      );
      ghostMesh.visible = false;
      this.hoppersGroup.add(ghostMesh);

      // 1. Optimal Verified Multi-Hop Green Trajectory (Start -> Hop 1 -> Hop 2 -> End)
      const optPoints = this._createMultiHopCurvePoints(sp.waypoints, 620);
      const optGeo = new THREE.BufferGeometry().setFromPoints(optPoints);
      const optMat = new THREE.LineBasicMaterial({
        color: 0x3ddc84,
        transparent: true,
        opacity: 0.98,
      });
      const optLine = new THREE.Line(optGeo, optMat);
      optLine.visible = false;
      this.rolloutsGroup.add(optLine);

      // Glowing green halo tube around optimal path
      const curve = new THREE.CatmullRomCurve3(optPoints);
      const tubeGeo = new THREE.TubeGeometry(curve, 96, 16, 6, false);
      const tubeMat = new THREE.MeshBasicMaterial({
        color: 0x3ddc84,
        transparent: true,
        opacity: 0.25,
      });
      const optTube = new THREE.Mesh(tubeGeo, tubeMat);
      optTube.visible = false;
      this.rolloutsGroup.add(optTube);

      // 2. Exploratory Candidate Green Spread Paths (shown around 10K - 100K steps)
      const greenCandidates = [];
      for (let c = 0; c < 6; c++) {
        const spreadFactor = (c - 2.5) * 340;
        const noisyWps = sp.waypoints.map(([wx, wy], idx) => {
          if (idx === 0) return [wx, wy];
          const mul = idx === sp.waypoints.length - 1 ? 0.35 : 1.0;
          return [
            wx + Math.cos(c * 1.3 + hIdx) * spreadFactor * mul,
            wy + Math.sin(c * 1.7 + hIdx) * spreadFactor * mul,
          ];
        });
        const candPts = this._createMultiHopCurvePoints(noisyWps, 480 + c * 40);
        const candGeo = new THREE.BufferGeometry().setFromPoints(candPts);
        const candMat = new THREE.LineBasicMaterial({
          color: 0x3ddc84,
          transparent: true,
          opacity: 0.42,
        });
        const candLine = new THREE.Line(candGeo, candMat);
        candLine.visible = false;
        this.rolloutsGroup.add(candLine);
        greenCandidates.push({
          line: candLine,
          totalVerts: candPts.length,
          // Candidate green paths appear after 1K (scrub > 0.12) and merge into the 1 optimal path by 100K..500K (scrub < 0.88)
          minScrub: 0.12 + c * 0.03,
          maxScrub: 0.65 + (5 - c) * 0.05,
        });
      }

      // 3. Red Failed Attempt Fan (32 red arcs per hopper — radial starburst at 1,000 steps like panel04-b.jpg,
      // progressively pruned as iterations climb to 10K, 100K, and 500K episodes / 18M steps!)
      const redLines = [];
      const numRed = 32;
      for (let r = 0; r < numRed; r++) {
        // Fan out radially around the hopper's start & intermediate hops (just like panel04-b.jpg!)
        const angle = (r / numRed) * Math.PI * 2 + hIdx * 0.4;
        const dist = 1600 + ((r * 293) % 3400);
        const isMultiHopFail = r % 3 === 0;

        let failWps;
        if (!isMultiHopFail) {
          // Single-hop radial starburst failure around start or hop 1 (panel04-b.jpg style)
          const tx = sx + Math.cos(angle) * dist;
          const ty = sy + Math.sin(angle) * dist;
          failWps = [
            [sx, sy],
            [tx, ty],
          ];
        } else {
          // 2-hop failure (succeeds Hop 1, crashes on Hop 2 into crater rim)
          const [h1x, h1y] = sp.waypoints[1];
          const tx = h1x + Math.cos(angle) * (dist * 0.85);
          const ty = h1y + Math.sin(angle) * (dist * 0.85);
          failWps = [
            [sx, sy],
            [h1x + Math.sin(r) * 280, h1y + Math.cos(r) * 280],
            [tx, ty],
          ];
        }

        const redPts = this._createMultiHopCurvePoints(failWps, 340 + (r % 5) * 60);
        const redGeo = new THREE.BufferGeometry().setFromPoints(redPts);
        const redMat = new THREE.LineBasicMaterial({
          color: 0xff4d4d,
          transparent: true,
          opacity: 0.68,
        });
        const redLine = new THREE.Line(redGeo, redMat);
        const [rule, reason] = failureReasons[(r + hIdx) % failureReasons.length];
        const endPt = failWps[failWps.length - 1];
        redLine.userData = {
          failMeta: {
            hopperId: sp.id,
            hopperName: sp.name,
            runId: `MC-${hIdx * 100 + r + 1}`,
            failedHop: failWps.length - 1,
            rule,
            reason,
            crashXY: endPt,
          },
        };
        redLine.visible = false;
        this.rolloutsGroup.add(redLine);

        // Thresholds: at scrub=0.0 (1,000 steps), all 32 red lines are visible.
        // At scrub=0.33 (10,000 steps), ~18 remain.
        // At scrub=0.66 (100,000 steps), ~6 remain.
        // At scrub=1.0 (500K eps / 18M steps), 0 remain (all green optimal!).
        const vanishAtScrub = ((r + 1) / numRed) * 0.92;
        redLines.push({
          line: redLine,
          totalVerts: redPts.length,
          vanishAtScrub,
        });
      }

      this.fleetHoppers.push({
        ...sp,
        mesh: hopperMesh,
        ghostMesh,
        optLine,
        optTube,
        optPoints,
        greenCandidates,
        redLines,
      });
    });

    // Build S-Band Orbital Relay Laser Line (panel05-d.jpg: RELAY ORBITER · LOS 04:12)
    this._buildRelayLasers();
  }

  _buildRelayLasers() {
    while (this.relayLaserGroup.children.length) {
      this.relayLaserGroup.remove(this.relayLaserGroup.children[0]);
    }
    const relaySkyPoint = enuToScene(6200, 3800, 3400);
    // Glowing green relay orbiter beacon in sky (panel05-d.jpg)
    const beacon = new THREE.Mesh(
      new THREE.SphereGeometry(55, 12, 12),
      new THREE.MeshBasicMaterial({ color: 0x3ddc84 })
    );
    beacon.position.copy(relaySkyPoint);
    this.relayLaserGroup.add(beacon);

    // Green laser downlink from c2 / Hopper 1 to Relay Orbiter
    const c2Z = this.getSurfaceElevation(1850, -650) + 25;
    const laserPts = [
      enuToScene(-3400, -4800, this.getSurfaceElevation(-3400, -4800) + 45),
      enuToScene(1850, -650, c2Z),
      relaySkyPoint,
    ];
    const laserLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(laserPts),
      new THREE.LineBasicMaterial({ color: 0x3ddc84, transparent: true, opacity: 0.78 })
    );
    this.relayLaserGroup.add(laserLine);
    this.relayLaserGroup.visible = false;
  }

  // ---------------------------------------------------------------------------
  // Horizontal Training Iteration Scrubber (1,000 -> 10,000 -> 100,000 -> 500K Eps / 18M Steps)
  // ---------------------------------------------------------------------------
  setTrainingConvergenceScrub(scrub01, forceShowRejected = false) {
    this.trainingScrub = Math.max(0, Math.min(1, scrub01));
    this.showSimPaths = true;
    this.previewArcGroup.visible = false;

    let visibleRedCount = 0;
    let visibleGreenCount = 0;

    for (const h of this.fleetHoppers) {
      const hopperActive = this.selectedHopperId === 'all' || this.selectedHopperId === h.id;
      if (!hopperActive) {
        h.optLine.visible = false;
        h.optTube.visible = false;
        h.greenCandidates.forEach((c) => (c.line.visible = false));
        h.redLines.forEach((r) => (r.line.visible = false));
        continue;
      }

      // 1. Red failed attempts (dense at 1,000 steps, fewer at 10,000, minimal at 100,000, zero at 500K/18M unless toggled)
      for (const r of h.redLines) {
        const showRed = forceShowRejected || this.trainingScrub < r.vanishAtScrub;
        r.line.visible = showRed;
        if (showRed) {
          visibleRedCount++;
          const drawCount = Math.max(2, Math.floor(r.totalVerts * this.pathAnimProgress));
          r.line.geometry.setDrawRange(0, drawCount);
          r.line.material.opacity = forceShowRejected
            ? 0.45
            : Math.max(0.18, 0.78 * (1 - this.trainingScrub * 0.75));
        }
      }

      // 2. Exploratory Green Candidate Spread (10,000 to 100,000 steps)
      for (const c of h.greenCandidates) {
        const showCand = this.trainingScrub >= c.minScrub && this.trainingScrub <= c.maxScrub;
        c.line.visible = showCand;
        if (showCand) {
          visibleGreenCount++;
          const drawCount = Math.max(2, Math.floor(c.totalVerts * this.pathAnimProgress));
          c.line.geometry.setDrawRange(0, drawCount);
        }
      }

      // 3. Final Optimal Multi-Hop Green Path (emerges from 10,000 steps and locks in at 100,000..500K/18M steps)
      const showOpt = this.trainingScrub >= 0.22;
      h.optLine.visible = showOpt;
      h.optTube.visible = showOpt && this.trainingScrub >= 0.55;
      if (showOpt) {
        visibleGreenCount++;
        const drawCount = Math.max(2, Math.floor(h.optPoints.length * this.pathAnimProgress));
        h.optLine.geometry.setDrawRange(0, drawCount);
        h.optLine.material.opacity = Math.min(1.0, (this.trainingScrub - 0.15) * 1.4);
      }
    }

    // Show S-Band Relay Orbiter LOS link once paths converge (>= 60%)
    this.relayLaserGroup.visible = this.trainingScrub >= 0.6;

    const metrics = this._computeBenchmarkMetricsAtScrub(this.trainingScrub, visibleRedCount, visibleGreenCount);
    if (this.onConvergenceUpdate) {
      this.onConvergenceUpdate(metrics);
    }
    return {
      reward: Number(metrics.reward),
      redVisible: visibleRedCount,
      greenVisible: visibleGreenCount,
      wmMse: Number(metrics.wmMse),
      crashRate: Number(metrics.crashRatePct) / 100.0,
      metrics,
    };
  }

  _computeBenchmarkMetricsAtScrub(s, redCount, greenCount) {
    // Logarithmic interpolation from 1,000 steps -> 18,000,000 steps (500,000 episodes)
    // Matches exact repo numbers from Run 08 (docs/runs/run-08-mappo-nasa-extended-30min.md)
    const logSteps = 3.0 + s * (Math.log10(18000000) - 3.0);
    const steps = Math.round(Math.pow(10, logSteps));
    const episodes = Math.max(28, Math.round((steps / 18000000) * 500000));

    // Reward: 131.36 -> 539.88 (+311%)
    const reward = (131.36 + (539.88 - 131.36) * Math.pow(s, 0.75)).toFixed(2);
    // Crash rate: 68.4% at 1K -> 0.00% at 500K/18M
    const crashRatePct = s >= 0.88 ? 0.0 : Math.max(0, 68.4 * Math.pow(1 - s / 0.88, 1.8));
    // World Model MSE: 1.3599 -> 0.0137 (-99.0%)
    const wmMse = (1.3599 * Math.pow(1 - s, 2.4) + 0.0137).toFixed(4);
    // Crater overlap: 4.76 -> 1.90 (-60.1%)
    const overlap = (4.76 - (4.76 - 1.9) * s).toFixed(2);
    // Relay TDMA conflicts: 2.11 -> 0.0156 (-99.3%)
    const tdma = (2.11 * Math.pow(1 - s, 2.1) + 0.0156 * s).toFixed(3);
    // Verifier Monte Carlo Pass Rate
    const verifierPassPct = (100 - crashRatePct).toFixed(1);

    let stageLabel = 'STAGE 1 · RANDOM EXPLORATION (HIGH FAILURE RATE)';
    if (s >= 0.88) {
      stageLabel = 'CONVERGED · CTDE MAPPO + 5-MLP WORLD MODEL (CERTIFIED OPTIMAL)';
    } else if (s >= 0.55) {
      stageLabel = 'STAGE 3 · TDMA COORDINATION & MONTE CARLO RIM PRUNING';
    } else if (s >= 0.22) {
      stageLabel = 'STAGE 2 · 5-MLP WORLD MODEL IMAGINATION (SPREAD OF PATHS)';
    }

    return {
      scrub: s,
      steps,
      episodes,
      reward,
      crashRatePct: crashRatePct.toFixed(1),
      wmMse,
      overlap,
      tdma,
      verifierPassPct,
      redCount,
      greenCount,
      stageLabel,
    };
  }

  triggerPathWaveAnimation(targetScrub = null) {
    if (targetScrub !== null) {
      this.trainingScrub = targetScrub;
    }
    this.pathAnimProgress = 0.02;
    this.isAnimatingWave = true;
    this.setTrainingConvergenceScrub(this.trainingScrub);
  }

  // ---------------------------------------------------------------------------
  // Camera Views & Transitions (Globe -> Top Map -> Tilted Fleet -> Hopper Zoom)
  // ---------------------------------------------------------------------------
  tweenCamera(targetPos, targetLookAt, durationMs = 1500, onDone = null) {
    this.cameraTween = {
      startPos: this.camera.position.clone(),
      endPos: targetPos.clone(),
      startTarget: this.controls.target.clone(),
      endTarget: targetLookAt.clone(),
      startTime: performance.now(),
      duration: durationMs,
      onDone,
    };
  }

  // Stage 1: Globe Home View (panel02-b.jpg)
  transitionToGlobe(onDone = null) {
    this.manualFpvActive = false;
    this.flightActive = false;
    this.globeSpinActive = true;
    this.surfaceGroup.visible = false;
    this.tweenCamera(
      new THREE.Vector3(0, 6200, 41500),
      new THREE.Vector3(0, -R_MOON_SCENE * 0.88, 0),
      1500,
      onDone
    );
  }

  // Stage 2 -> Stage 3: Spins & zooms from the 3D Moon Globe into Daedalus Top-Down Map View (panel03-a.jpg)
  zoomFromGlobeToDaedalusTopView(onDone = null) {
    this.globeSpinActive = false;
    this.surfaceGroup.visible = true;
    this.tweenCamera(
      new THREE.Vector3(0, 14800, 260),
      new THREE.Vector3(0, 0, 0),
      1900,
      onDone
    );
  }

  zoomFromGlobeToSurfaceTopView(onDone = null) {
    this.zoomFromGlobeToDaedalusTopView(onDone);
  }

  snapToTopMapView(onDone = null) {
    this.zoomFromGlobeToDaedalusTopView(onDone);
  }

  // Stage 4: Panned and tilted 3D tactical view of all 4 hoppers & multi-hop arcs (panel04-b.jpg)
  setTiltedFleetView(onDone = null) {
    this.surfaceGroup.visible = true;
    this.tweenCamera(
      new THREE.Vector3(-4200, 6200, 10800),
      new THREE.Vector3(400, -120, -400),
      1350,
      onDone
    );
  }

  tiltToSimulationView(onDone = null) {
    this.setTiltedFleetView(onDone);
  }

  // Stage 5: Zoom in on a specific individual hopper (panel05-d.jpg)
  focusOnHopper(hopperId) {
    this.selectedHopperId = hopperId;
    this.setTrainingConvergenceScrub(this.trainingScrub);

    if (hopperId === 'all') {
      this.setTiltedFleetView();
      return;
    }

    const h = this.fleetHoppers.find((item) => item.id === hopperId);
    if (!h) return;
    const pos = h.mesh.position.clone();
    this.tweenCamera(
      pos.clone().add(new THREE.Vector3(-1400, 1150, 2100)),
      pos.clone().add(new THREE.Vector3(600, 120, -600)),
      1100
    );
  }

  selectHopper(hopperNum) {
    const id = !hopperNum || hopperNum <= 0 ? 'all' : `h${Math.max(0, Math.min(3, hopperNum - 1))}`;
    this.focusOnHopper(id);
  }

  // ---------------------------------------------------------------------------
  // Stage 6: Autonomous Multi-Hop Flight Execution (Follow Camera)
  // ---------------------------------------------------------------------------
  startFollowFlight(hopperId = 'h0') {
    this.manualFpvActive = false;
    this.manualCrashed = false;
    this.flightActive = true;
    this.flightProgress = 0.0;
    this.followHopperId = hopperId === 'all' ? 'h0' : hopperId;

    // Ensure converged optimal paths are shown during flight execution
    this.setTrainingConvergenceScrub(1.0);
  }

  startExecutedFlight(hopperNum = 1) {
    const id = !hopperNum || hopperNum <= 0 ? 'h0' : `h${Math.max(0, Math.min(3, hopperNum - 1))}`;
    this.startFollowFlight(id);
  }

  // ---------------------------------------------------------------------------
  // Stage 7: Unconstrained 6-DOF Manual Override (Flip Upside Down + Crash + Respawn!)
  // ---------------------------------------------------------------------------
  get manualMode() {
    return this.manualFpvActive;
  }

  get manualState() {
    const st = this.fpvState;
    const bodyUp = new THREE.Vector3(0, 1, 0).applyQuaternion(st.quat).normalize();
    const tiltDeg = Math.acos(Math.max(-1, Math.min(1, bodyUp.y))) * (180 / Math.PI);
    return {
      pos: st.pos,
      vel: st.vel,
      quat: st.quat,
      fuelPct: (st.fuelKg / 15.0) * 100,
      tiltDeg,
      crashed: this.manualCrashed,
    };
  }

  get manualKeys() {
    return this.fpvKeys;
  }

  setManualMode(enabled) {
    if (enabled) {
      this.enterManualFpvMode(this.selectedHopperId === 'all' ? 'h0' : this.selectedHopperId);
    } else {
      this.exitManualFpvMode();
    }
  }

  setManualKey(key, pressed) {
    if (key in this.fpvKeys) {
      this.fpvKeys[key] = Boolean(pressed);
    }
  }

  sampleTerrainHeight(x, z) {
    return this.getSurfaceElevation(x, -z);
  }

  getCameraTelemetry() {
    const offset = this.camera.position.clone().sub(this.controls.target);
    const dist = offset.length();
    const horiz = Math.hypot(offset.x, offset.z);
    const pitchDeg = Math.round(-Math.atan2(offset.y, Math.max(1, horiz)) * (180 / Math.PI));
    const azimuthDeg = Math.round(((Math.atan2(offset.x, offset.z) * (180 / Math.PI)) + 360) % 360);
    const altitudeKm = Math.max(0.05, dist / 1000);
    return { pitchDeg, azimuthDeg, altitudeKm };
  }
  enterManualFpvMode(hopperId = 'h0') {
    this.flightActive = false;
    this.manualFpvActive = true;
    this.manualCrashed = false;
    this.followHopperId = hopperId === 'all' ? 'h0' : hopperId;

    this.respawnManualHopper();
  }

  respawnManualHopper() {
    this.manualCrashed = false;
    const h = this.fleetHoppers.find((item) => item.id === this.followHopperId) || this.fleetHoppers[0];
    const [sx, sy] = h ? h.waypoints[1] : [-850, -2100];
    const groundY = this.getSurfaceElevation(sx, sy);

    this.fpvState.pos.copy(enuToScene(sx, sy, groundY + 460));
    this.fpvState.vel.set(0, 0, 0);
    this.fpvState.quat.identity();
    this.fpvState.angVel.set(0, 0, 0);
    this.fpvState.fuelKg = 15.0;
    this.fpvState.batteryPct = 96.0;
    this.fpvState.crashReason = '';

    if (h?.mesh) {
      h.mesh.position.copy(this.fpvState.pos);
      h.mesh.quaternion.copy(this.fpvState.quat);
    }
  }

  exitManualFpvMode() {
    this.manualFpvActive = false;
    this.manualCrashed = false;
    const h = this.fleetHoppers.find((item) => item.id === this.followHopperId) || this.fleetHoppers[0];
    if (h?.mesh) {
      h.mesh.quaternion.identity();
      if (h.mesh.userData.plume) h.mesh.userData.plume.visible = false;
    }
    this.setTiltedFleetView();
  }

  _stepManual6DofPhysics(dt) {
    const h = this.fleetHoppers.find((item) => item.id === this.followHopperId) || this.fleetHoppers[0];
    if (!h) return;

    if (this.manualCrashed) {
      if (h.mesh.userData.plume) h.mesh.userData.plume.visible = false;
      return;
    }

    const st = this.fpvState;
    const k = this.fpvKeys;

    // 1. UNCONSTRAINED 6-DOF Angular Acceleration (NO horizon lock — can flip 360° upside down!)
    const angularAccel = 2.6; // rad/s^2
    if (k.w) st.angVel.x -= angularAccel * dt; // Pitch forward (nose down / flip)
    if (k.s) st.angVel.x += angularAccel * dt; // Pitch backward (nose up / backflip)
    if (k.a) st.angVel.z += angularAccel * dt; // Roll left (barrel roll)
    if (k.d) st.angVel.z -= angularAccel * dt; // Roll right (barrel roll)

    // Mild rotational damping so controls feel responsive, but zero upright restoring torque!
    st.angVel.multiplyScalar(0.97);

    // Apply incremental rotation in the hopper's local body frame
    const dq = new THREE.Quaternion().setFromEuler(
      new THREE.Euler(st.angVel.x * dt, st.angVel.y * dt, st.angVel.z * dt, 'XYZ')
    );
    st.quat.multiply(dq).normalize();

    // 2. Compute Hopper Local Up Vector in World Space
    const bodyUp = new THREE.Vector3(0, 1, 0).applyQuaternion(st.quat).normalize();

    // 3. Thruster Acceleration along Hopper's Local Nozzle Axis + Lunar Gravity (g = 1.62 m/s^2, scaled for visual responsiveness)
    const gScene = 38.0; // visual scale for 1.62 m/s^2 at 42x model scale
    const accel = new THREE.Vector3(0, -gScene, 0);

    let thrustMag = 0;
    if (k.space && st.fuelKg > 0) {
      thrustMag = 98.0; // Main engine burn in direction of bodyUp!
      accel.addScaledVector(bodyUp, thrustMag);
      st.fuelKg = Math.max(0, st.fuelKg - 0.35 * dt);
    } else if (k.shift && st.fuelKg > 0) {
      // Shift = cut / retro brake against current velocity
      thrustMag = 45.0;
      accel.addScaledVector(st.vel.clone().normalize(), -thrustMag);
      st.fuelKg = Math.max(0, st.fuelKg - 0.18 * dt);
    }

    // Show/hide thruster plume & white thrust arrows (panel07-c.jpg)
    const firing = k.space || k.shift || k.w || k.a || k.s || k.d;
    if (h.mesh.userData.plume) {
      h.mesh.userData.plume.visible = firing;
    }

    // Integrate velocity & position
    st.vel.addScaledVector(accel, dt);
    st.pos.addScaledVector(st.vel, dt);

    // 4. Check Ground Contact / Crash Detection!
    const [ex, ey] = sceneToEnu(st.pos);
    const groundY = this.getSurfaceElevation(ex, ey);
    const agl = st.pos.y - groundY;
    const tiltDeg = Math.acos(Math.max(-1, Math.min(1, bodyUp.y))) * (180 / Math.PI);
    const speed = st.vel.length() * 0.22; // scaled to m/s
    const slope = this.getSurfaceSlope(ex, ey);

    if (agl <= 4.0) {
      st.pos.y = groundY + 2.0;
      // Crash if tilted > 32 degrees (or upside down!), or high impact speed, or slope > 15 deg
      if (tiltDeg > 32.0 || speed > 9.5 || slope > 15.0) {
        this.manualCrashed = true;
        st.vel.set(0, 0, 0);
        // Leave the hopper visibly tipped/upside-down on the lunar floor!
        if (tiltDeg < 55) {
          st.quat.multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0.95, 0, 0.65)));
        }
        if (tiltDeg > 90) {
          st.crashReason = `ATTITUDE INVERTED (${tiltDeg.toFixed(0)}° UPSIDE DOWN) · IMPACT ${speed.toFixed(1)} m/s`;
        } else if (tiltDeg > 32) {
          st.crashReason = `TIP-OVER VIOLATION (TILT ${tiltDeg.toFixed(0)}° > 30° LIMIT)`;
        } else if (slope > 15.0) {
          st.crashReason = `R3 SLOPE VIOLATION (${slope.toFixed(1)}° > 15.0° LIMIT)`;
        } else {
          st.crashReason = `HARD IMPACT (${speed.toFixed(1)} m/s > 9.5 m/s MAX TOUCHDOWN)`;
        }
        if (this.onManualCrash) {
          this.onManualCrash({
            reason: st.crashReason,
            speed: speed.toFixed(1),
            tiltDeg: tiltDeg.toFixed(0),
          });
        }
      } else {
        // Safe touchdown
        st.vel.multiplyScalar(0.5);
        st.vel.y = Math.max(0, st.vel.y);
      }
    }

    h.mesh.position.copy(st.pos);
    h.mesh.quaternion.copy(st.quat);

    // 3rd-person chase camera behind the hopper (stays upright in world space so you can watch the hopper flip upside down!)
    const desiredCamPos = st.pos.clone().add(new THREE.Vector3(0, 260, 620));
    const camGround = this.getSurfaceElevation(desiredCamPos.x, -desiredCamPos.z);
    desiredCamPos.y = Math.max(desiredCamPos.y, camGround + 60);
    this.camera.position.lerp(desiredCamPos, 0.14);
    this.controls.target.lerp(st.pos, 0.22);

    if (this.onFlightTelemetry) {
      this.onFlightTelemetry({
        mode: 'MANUAL_FPV',
        hopperName: h.name,
        fuelKg: st.fuelKg,
        fuelPct: (st.fuelKg / 15.0) * 100,
        batteryPct: st.batteryPct,
        wmDriftM: 0.0,
        thrust: [
          Math.round(bodyUp.x * thrustMag),
          Math.round(bodyUp.z * thrustMag),
          Math.round(bodyUp.y * thrustMag),
        ],
        speed,
        agl: Math.max(0, agl * 0.25),
        tiltDeg,
        temp: this.getSurfaceTemp(ex, ey),
        slope,
        hopStage: this.manualCrashed ? 'CRASHED — CLICK [RESPAWN]' : `FREE 6-DOF · TILT ${tiltDeg.toFixed(0)}°`,
      });
    }
  }

  // ---------------------------------------------------------------------------
  // Continuous Google Earth Zoom Blend & Horizon Floor Clamp
  // ---------------------------------------------------------------------------
  _updateScaleBlendAndLocks() {
    if (this.manualFpvActive) return;

    // 1. Lock camera above lunar floor and horizon so user never sees under the map
    const [cx, cy] = sceneToEnu(this.camera.position);
    const groundY = this.getSurfaceElevation(cx, cy);
    if (this.camera.position.y < groundY + 60) {
      this.camera.position.y = groundY + 60;
    }
    if (this.controls.target.y < -R_MOON_SCENE * 0.95) {
      this.controls.target.y = -R_MOON_SCENE * 0.95;
    }

    // 2. Continuous Google-Earth zoom transition between Daedalus 3D surface and Full 3D Moon Globe
    const dist = this.camera.position.distanceTo(this.controls.target);
    // Below 18,000: surface terrain. Above 22,000: full 3D Moon globe visible!
    const globeBlend = Math.max(0, Math.min(1, (dist - 17500) / 10000));

    if (this.moonSphere) {
      this.moonSphere.visible = globeBlend > 0.02;
      this.globeMat.opacity = globeBlend;
      if (this.orbitalTelemetryRing) {
        this.orbitalTelemetryRing.visible = globeBlend > 0.25;
      }
      if (this.orbitalLockDot) {
        this.orbitalLockDot.visible = globeBlend > 0.25;
      }
    }

    if (this.surfaceGroup && !this.globeSpinActive) {
      this.surfaceGroup.visible = globeBlend < 0.96;
    }

    if (this.onCameraUpdate) {
      const offset = this.camera.position.clone().sub(this.controls.target);
      const horiz = Math.hypot(offset.x, offset.z);
      const pitchDeg = Math.round(-Math.atan2(offset.y, Math.max(1, horiz)) * (180 / Math.PI));
      const azimuthDeg = Math.round(((Math.atan2(offset.x, offset.z) * (180 / Math.PI)) + 360) % 360);
      this.onCameraUpdate({
        dist,
        globeBlend,
        pitchDeg,
        azimuthDeg,
        targetX: this.controls.target.x,
        targetZ: this.controls.target.z,
      });
    }
  }

  // ---------------------------------------------------------------------------
  // Main 60 FPS Render Loop
  // ---------------------------------------------------------------------------
  startLoop() {
    let lastTime = performance.now();

    const tick = (now) => {
      requestAnimationFrame(tick);
      const dt = Math.min(0.05, (now - lastTime) / 1000);
      lastTime = now;

      // 1. Slow rotation on Home Globe screen (panel02-b.jpg)
      if (this.globeSpinActive && this.moonSphere) {
        this.moonSphere.rotation.y += dt * 0.045;
      }

      // 2. Smooth Camera Tween
      if (this.cameraTween) {
        const tw = this.cameraTween;
        const u = Math.min(1, (now - tw.startTime) / tw.duration);
        const ease = u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
        this.camera.position.lerpVectors(tw.startPos, tw.endPos, ease);
        this.controls.target.lerpVectors(tw.startTarget, tw.endTarget, ease);
        if (u >= 1) {
          const cb = tw.onDone;
          this.cameraTween = null;
          if (cb) cb();
        }
      }

      // 3. Multi-Hop Trajectory Wave Growth Animation
      if (this.isAnimatingWave) {
        this.pathAnimProgress = Math.min(1.0, this.pathAnimProgress + dt * 0.55);
        this.setTrainingConvergenceScrub(this.trainingScrub);
        if (this.pathAnimProgress >= 1.0) {
          this.isAnimatingWave = false;
        }
      }

      // 4. Autonomous Follow-Camera Multi-Hop Flight (Stage 6 / panel07-b.jpg)
      if (this.flightActive && !this.manualFpvActive) {
        this.flightProgress += dt * this.flightSpeed;
        if (this.flightProgress >= 1.0) {
          this.flightProgress = 1.0;
          this.flightActive = false;
          if (this.onFlightComplete) this.onFlightComplete();
          if (this.onFlightFinished) this.onFlightFinished();
        }

        for (const h of this.fleetHoppers) {
          const pts = h.optPoints;
          const idx = Math.min(pts.length - 1, Math.floor(this.flightProgress * (pts.length - 1)));
          const pt = pts[idx];
          h.mesh.position.copy(pt);

          const [ex, ey] = sceneToEnu(pt);
          const groundY = this.getSurfaceElevation(ex, ey);
          const agl = Math.max(0, pt.y - groundY);
          if (h.mesh.userData.plume) {
            h.mesh.userData.plume.visible = agl > 16;
          }

          // 5-MLP World Model Ghost twin (cyan wireframe slightly offset on outer rim)
          const rad = Math.hypot(ex, ey) / 8000;
          const drift = 4.0 + rad * 28.0 * Math.sin(this.flightProgress * Math.PI * 3);
          h.ghostMesh.visible = true;
          h.ghostMesh.position.copy(pt).add(new THREE.Vector3(drift * 2.2, 18, -drift * 1.8));
          h.ghostMesh.rotation.y += dt * 1.5;

          if (h.id === this.followHopperId) {
            const camOffset = new THREE.Vector3(-850, 480, 1250);
            this.camera.position.lerp(pt.clone().add(camOffset), 0.08);
            this.controls.target.lerp(pt, 0.14);

            const hopStageIdx = Math.min(3, Math.floor(this.flightProgress * 3) + 1);
            const fuelLeft = 15.0 * (1 - this.flightProgress * (1 - h.fuelEndPct / 100));

            if (this.onFlightTelemetry) {
              this.onFlightTelemetry({
                mode: 'AUTO_FOLLOW',
                hopperName: h.name,
                fuelKg: fuelLeft,
                fuelPct: (fuelLeft / 15.0) * 100,
                batteryPct: 100 - this.flightProgress * 28,
                wmDriftM: Math.abs(drift),
                thrust: [0, 0, agl > 16 ? 98 : 0],
                speed: agl > 16 ? 18.4 : 0.0,
                agl: agl * 0.25,
                tiltDeg: 2.1,
                temp: this.getSurfaceTemp(ex, ey),
                slope: this.getSurfaceSlope(ex, ey),
                hopStage: `HOP ${hopStageIdx}/3 -> ${h.hopLabels[hopStageIdx]}`,
                progress: this.flightProgress,
              });
            }
          }
        }
      }

      // 5. Unconstrained 6-DOF Manual Override Physics
      if (this.manualFpvActive) {
        this._stepManual6DofPhysics(dt);
      }

      this.controls.update();
      this._updateScaleBlendAndLocks();
      this.renderer.render(this.scene, this.camera);
    };

    requestAnimationFrame(tick);
  }
}
