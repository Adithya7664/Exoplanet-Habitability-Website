import numpy as np
import streamlit as st

# ── helpers ───────────────────────────────────────────────────────────────────
def _g(d: dict, key: str, default: float) -> float:
    try:
        f = float(d.get(key))
        return default if f != f else f  # f != f is True only for NaN
    except (TypeError, ValueError):
        return default
        

def _is_gas_giant(planet: dict) -> bool:
    return (_g(planet, "pl_radj", 0.0) > 0.2 or _g(planet, "pl_bmassj", 0.0) > 0.05)

def _planet_seed(planet: dict) -> float:
    return float(sum(ord(c) for c in str(planet.get("pl_name", "X"))) % 997) / 997.0

def _roughness_from_age(planet: dict, has_atm: bool) -> float:
    age = _g(planet, "st_age", 3.0)
    if has_atm:
        return float(np.interp(age, [0, 1, 3, 7, 12], [0.90, 0.75, 0.55, 0.35, 0.20]))
    return float(np.interp(age, [0, 2, 5, 10, 12], [0.85, 0.70, 0.65, 0.68, 0.72]))

def _snow_line(temp: float) -> float:
    if temp > 200: return 0.38
    if temp < 50: return 0.05
    return float(np.interp(temp, [80, 110, 150, 190, 200], [0.05, 0.15, 0.28, 0.37, 0.38]))

def _glow_rgb(oxygen: float) -> tuple[float, float, float]:
    o_pts = [0,   7,   14,  21,  38,  55,  70]
    r_pts = [190, 170, 130,  50,  60, 190, 200]
    g_pts = [ 30,  30,  50, 100, 150, 155, 165]
    b_pts = [ 30,  60, 160, 200, 160,  60,  40]
    o = float(np.clip(oxygen, 0, 70))
    return (float(np.interp(o, o_pts, r_pts)) / 255,
            float(np.interp(o, o_pts, g_pts)) / 255,
            float(np.interp(o, o_pts, b_pts)) / 255)

def _surface_colors(planet: dict, pressure: float | None = None, oxygen: float | None = None) -> tuple[str, str]:
    temp    = _g(planet, "pl_eqt",   255.0)
    esi     = _g(planet, "esi",      0.0)
    insol   = _g(planet, "pl_insol", 1.0)
    st_teff = _g(planet, "st_teff",  5778.0)
    st_met  = _g(planet, "st_met",   0.0)
    sw  = float(np.clip((5778 - st_teff) / 3000, -1, 1))          # +1=red dwarf, -1=hot star
    met = float(np.clip(st_met / 0.5, -1, 1))                      # +1=metal-rich
    ins = float(np.clip(np.log10(max(insol, 0.001)) / 2, -1, 1))  # +1=high insolation

    has_atm   = pressure is not None and pressure > 0.05
    has_water = (200 < temp < 390) and (esi > 0.4 or (has_atm and 260 < temp < 380))
    has_veg   = (oxygen is not None and oxygen > 10) or (has_water and esi > 0.65 and 270 < temp < 340)

    if has_water:
        ro = int(np.clip(12 + sw * 18, 5, 35))
        go = int(np.clip(62 + sw * 8,  50, 80))
        bo = int(np.clip(148 - sw * 25, 100, 170))
        dark_hex = f"#{ro:02x}{go:02x}{bo:02x}"
        if temp < 255:
            r = int(np.clip(82 + sw * 15, 50, 105))
            g, b = 100, 85
        elif has_veg:
            r = int(np.clip(38 + sw * 25, 20, 80))
            g = int(np.clip(112 - sw * 15, 75, 135))
            b = int(np.clip(32 - sw * 8, 15, 55))
        elif temp < 310:
            r, g, b = 130, 110, 55
        else:
            r = int(np.clip(165 + sw * 20, 130, 200))
            g = int(np.clip(95 - sw * 10, 60, 115))
            b = 35
        return dark_hex, f"#{r:02x}{g:02x}{b:02x}"

    if temp > 700:
        r = int(np.clip(225 + met * 25, 195, 255))
        g = int(np.clip(45  + met * 30, 20, 90))
        return f"#{int(r*.28):02x}{int(g*.25):02x}00", f"#{r:02x}{g:02x}00"

    if temp > 400:
        r = int(np.clip(185 + sw * 40 + met * 25, 130, 240))
        g = int(np.clip(58  - sw * 12 + met * 18, 25, 105))
        b = int(np.clip(18  - sw * 8, 5, 38))
        return f"#{int(r*.38):02x}{int(g*.38):02x}{int(b*.38):02x}", f"#{r:02x}{g:02x}{b:02x}"

    if temp > 200:
        br = int(np.clip(158 + ins * 42 + sw * 28, 90, 218))
        bg = int(np.clip(88  + ins * 20 + met * 18, 45, 140))
        bb = int(np.clip(38  - sw * 10, 12, 68))
        return f"#{int(br*.35):02x}{int(bg*.35):02x}{int(bb*.35):02x}", f"#{br:02x}{bg:02x}{bb:02x}"

    if temp > 130:
        r = int(np.clip(88  - sw * 10, 55, 110))
        g = int(np.clip(82  - sw * 5,  62, 102))
        b = int(np.clip(78  + sw * 12, 52, 98))
        return "#181816", f"#{r:02x}{g:02x}{b:02x}"

    return "#c8dce8", "#f0f8ff"

def _gas_colors(planet: dict) -> tuple[str, str, str, int]:
    temp  = _g(planet, "pl_eqt",   500.0)
    massj = _g(planet, "pl_bmassj", 0.3)
    n = int(np.clip(round(massj * 3), 1, 5))
    if temp > 1500: return "#050002", "#3a0c1a", "#700a20", n
    if temp > 700:  return "#0d0200", "#7a2a08", "#d05020", n
    if temp > 350:  return "#7a3808", "#d88030", "#f0b050", n
    if temp > 100:  return "#060c1a", "#1a4878", "#4090c8", 2
    return               "#030810", "#1850a0", "#5090e0", 2

# ── shaders ───────────────────────────────────────────────────────────────────

_VERT_SURF = """
varying vec3 vNormal;
varying vec3 vPos;
void main() {
    vNormal = normalize(normalMatrix * normal);
    vPos = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
"""

_FRAG_SURF = """
varying vec3 vNormal;
varying vec3 vPos;
uniform vec3 uLight;
uniform vec3 uOcean;
uniform vec3 uSun;
uniform float uRoughness;
uniform float uSnowLine;
uniform float uWater;
uniform float uSeed;

float hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1); p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float hash1(float n) { return fract(sin(n) * 43758.5453); }
float noise(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash(i),hash(i+vec3(1,0,0)),f.x),mix(hash(i+vec3(0,1,0)),hash(i+vec3(1,1,0)),f.x),f.y),
        mix(mix(hash(i+vec3(0,0,1)),hash(i+vec3(1,0,1)),f.x),mix(hash(i+vec3(0,1,1)),hash(i+vec3(1,1,1)),f.x),f.y),f.z);
}
float fbm(vec3 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 6; i++) { v += a * noise(p); p *= 2.1; a *= 0.5; }
    return v;
}
float craters(vec3 p) {
    float ch = 0.0;
    for (int ci = 0; ci < 12; ci++) {
        float fi = float(ci) * 3.7 + uSeed * 97.3;
        float clat = hash1(fi*13.1+0.5)*1.8-0.9;
        float clon = hash1(fi*27.3+1.5)*6.28318;
        float cr = hash1(fi*41.7+2.5)*0.22+0.09;
        float csr = sqrt(max(0.0,1.0-clat*clat));
        vec3 cc = vec3(csr*cos(clon),clat,csr*sin(clon));
        float nd = length(p-cc)/cr;
        ch += -(1.0-smoothstep(0.0,0.70,nd))*0.5 + smoothstep(0.65,0.85,nd)*(1.0-smoothstep(0.85,1.4,nd))*0.6;
    }
    return ch;
}
float bumpAt(vec3 p, float crStr) {
    return fbm(p*3.0+2.1)*0.7 + fbm(p*7.0+5.3)*0.3 + craters(p)*crStr;
}
void main() {
    float wx = noise(vPos * 0.9 + vec3(uSeed * 2.1, 1.3, 0.7));
    float wy = noise(vPos * 0.9 + vec3(0.4, uSeed * 1.7, 2.3));
    float hColor = fbm((vPos + vec3(wx, wy, wx * 0.5) * 1.2) * 1.5 + vec3(uSeed * 3.1, uSeed * 1.7, uSeed * 2.3));
    float landness = uWater > 0.5
        ? smoothstep(uSnowLine - 0.05, uSnowLine + 0.05, hColor)
        : 1.0;

    float crStr = (1.0 - uRoughness) * 4.0 * landness;
    float hBump = bumpAt(vPos, crStr);

    float waterMix = step(0.5, uWater) * (1.0 - landness);
    vec3 col = mix(uLight, uOcean, waterMix);

    float bumpStr = uRoughness * mix(0.03, 0.12, landness);
    float eps = 0.012;
    float bX = bumpAt(vPos + vec3(eps, 0.0, 0.0), crStr) - hBump;
    float bY = bumpAt(vPos + vec3(0.0, eps, 0.0), crStr) - hBump;
    float bZ = bumpAt(vPos + vec3(0.0, 0.0, eps), crStr) - hBump;
    vec3 grad = vec3(bX, bY, bZ) / eps;
    vec3 projGrad = grad - dot(grad, vNormal) * vNormal;
    vec3 bumpN = normalize(vNormal - projGrad * bumpStr);

    float diff = max(dot(bumpN, normalize(uSun)), 0.0);
    col *= (0.12 + diff * 0.88);

    vec3 vd = normalize(cameraPosition - vPos);
    float specPow = mix(18.0, 48.0, landness);
    float specStr = mix(0.55, 0.18, landness);
    float spec = pow(max(dot(vd, reflect(-normalize(uSun), bumpN)), 0.0), specPow) * specStr;
    gl_FragColor = vec4(col + spec, 1.0);
}
"""

_VERT_CLOUD = """
varying vec3 vPos;
varying vec3 vNormal;
void main() {
    vPos = position;
    vNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
"""

_FRAG_CLOUD = """
varying vec3 vPos;
varying vec3 vNormal;
uniform vec3 uSun;
uniform float uCoverage;
uniform float uOpacity;

float hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1); p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float noise(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash(i),hash(i+vec3(1,0,0)),f.x),mix(hash(i+vec3(0,1,0)),hash(i+vec3(1,1,0)),f.x),f.y),
        mix(mix(hash(i+vec3(0,0,1)),hash(i+vec3(1,0,1)),f.x),mix(hash(i+vec3(0,1,1)),hash(i+vec3(1,1,1)),f.x),f.y),f.z);
}
float fbm(vec3 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) { v += a * noise(p); p *= 2.1; a *= 0.5; }
    return v;
}
void main() {
    float n = fbm(vPos * 2.5 + 0.5) * 0.55 + fbm(vPos * 5.0 + 3.3) * 0.45;
    float alpha = smoothstep(uCoverage, uCoverage + 0.15, n) * uOpacity;
    float diff = max(dot(vNormal, normalize(uSun)), 0.0);
    gl_FragColor = vec4(vec3(0.82 + diff * 0.18), alpha);
}
"""

_VERT_ATM = """
varying float intensity;
void main() {
    vec3 vN = normalize(normalMatrix * normal);
    intensity = pow(1.0 - dot(vN, vec3(0.0, 0.0, 1.0)), 5.0);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
"""

_FRAG_ATM = """
varying float intensity;
uniform vec3 uGlow;
uniform float uStrength;
void main() {
    gl_FragColor = vec4(uGlow, intensity * uStrength);
}
"""

_VERT_GAS = """
varying vec3 vPos;
varying vec3 vNormal;
void main() {
    vPos = position;
    vNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
"""

_FRAG_GAS = """
varying vec3 vPos;
varying vec3 vNormal;
uniform vec3  uSun;
uniform float uSeed;
uniform float uStormCount;
uniform float uTemp;

float hash1(float n) { return fract(sin(n) * 43758.5453); }
float hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1); p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float noise(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash(i),hash(i+vec3(1,0,0)),f.x),mix(hash(i+vec3(0,1,0)),hash(i+vec3(1,1,0)),f.x),f.y),
        mix(mix(hash(i+vec3(0,0,1)),hash(i+vec3(1,0,1)),f.x),mix(hash(i+vec3(0,1,1)),hash(i+vec3(1,1,1)),f.x),f.y),f.z);
}
float fbm(vec3 p) {
    float v=0.0,a=0.5;
    for(int i=0;i<4;i++){v+=a*noise(p);p*=2.1;a*=0.5;}
    return v;
}
void main() {
    // 3-stop palette: dark / mid / light — initialised with ice-giant defaults
    vec3 c0 = vec3(0.04, 0.18, 0.42);
    vec3 c1 = vec3(0.16, 0.35, 0.60);
    vec3 c2 = vec3(0.28, 0.52, 0.76);
    if      (uTemp > 1500.0) { c0 = vec3(0.01, 0.00, 0.00); c1 = vec3(0.22, 0.02, 0.06); c2 = vec3(0.44, 0.04, 0.13); }
    else if (uTemp >  700.0) { c0 = vec3(0.04, 0.01, 0.00); c1 = vec3(0.25, 0.08, 0.02); c2 = vec3(0.50, 0.18, 0.05); }
    else if (uTemp >  350.0) { c0 = vec3(0.20, 0.07, 0.01); c1 = vec3(0.55, 0.25, 0.06); c2 = vec3(0.95, 0.52, 0.22); }
    else if (uTemp >  100.0) { c0 = vec3(0.02, 0.05, 0.12); c1 = vec3(0.07, 0.18, 0.38); c2 = vec3(0.14, 0.32, 0.60); }

    // latitude + turbulence displacement
    float turb = (fbm(vPos * 1.8 + vec3(uSeed * 3.1, 0.5, uSeed * 1.7)) - 0.5) * 0.30;
    float lat  = vPos.y + turb;

    // sum of 3 sine frequencies — interference creates bands of varying widths
    float s = sin(lat * 7.0)                             * 0.50
            + sin(lat * 14.0 + uSeed * 5.8)             * 0.30
            + sin(lat * 29.0 + uSeed * 2.9)             * 0.15
            + (fbm(vPos * 6.0 + vec3(uSeed, uSeed * 1.3, uSeed * 0.7)) - 0.5) * 0.10;

    float t  = clamp(s * 0.5 + 0.5, 0.0, 1.0);
    vec3 col = mix(c0, c1, smoothstep(0.20, 0.50, t));
    col      = mix(col, c2, smoothstep(0.50, 0.80, t));

    // storms: distinct colour between dark and light
    vec3 stormCol = c0 * 0.30 + c2 * 0.70;
    for (int si = 0; si < 5; si++) {
        float fi   = float(si) + uSeed * 13.7;
        float act  = step(float(si) + 0.5, uStormCount);
        float slat = (hash1(fi * 17.3 + 1.1) - 0.5) * 1.4;
        float slon = hash1(fi * 31.7 + 2.3) * 6.2832;
        float srad = hash1(fi * 47.1 + 3.7) * 0.18 + 0.06;
        float csr  = sqrt(max(0.0, 1.0 - slat * slat));
        vec3  sc   = vec3(csr * cos(slon), slat, csr * sin(slon));
        float sm   = smoothstep(srad, srad * 0.45, length(vPos - sc)) * act;
        col = mix(col, stormCol, sm * 0.70);
    }

    float diff = max(dot(normalize(vNormal), normalize(uSun)), 0.0);
    col *= (0.18 + diff * 0.82);
    gl_FragColor = vec4(col, 1.0);
}
"""


# ── renderer ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=None)
def render_planet_3d(planet: dict, height: int = 420, pressure: float | None = None, oxygen: float | None = None) -> str:
    temp         = _g(planet, "pl_eqt", 255)
    esi          = _g(planet, "esi", 0.0)
    gas          = _is_gas_giant(planet)
    use_explicit = pressure is not None and oxygen is not None

    if gas:
        band_a, band_b, storm_hex, n_storms = _gas_colors(planet)
        seed = _planet_seed(planet)
        temp_gas = _g(planet, "pl_eqt", 500.0)
        if temp_gas > 700:
            atm_strength, gr, gg, gb = 0.30, 0.90, 0.40, 0.20  # fiery orange glow
        elif temp_gas > 350:
            atm_strength, gr, gg, gb = 0.25, 0.80, 0.55, 0.30  # warm orange
        else:
            atm_strength, gr, gg, gb = 0.22, 0.50, 0.70, 0.90  # blue for ice giants
        planet_block = f"""
const gasMat = new THREE.ShaderMaterial({{
    uniforms: {{
        uSun:        {{ value: sunDir.clone() }},
        uSeed:       {{ value: {seed:.4f} }},
        uStormCount: {{ value: {float(n_storms):.1f} }},
        uTemp:       {{ value: {temp_gas:.1f} }},
    }},
    vertexShader: gasVert, fragmentShader: gasFrag,
}});
scene.add(new THREE.Mesh(new THREE.SphereGeometry(1, 64, 64), gasMat));
"""
        cloud_block = ""
    else:
        dark_hex, light_hex = _surface_colors(planet, pressure, oxygen)
        has_atm = (use_explicit and pressure is not None and pressure > 0.05) or (not use_explicit and esi > 0.35)
        roughness = _roughness_from_age(planet, has_atm)
        if use_explicit:
            has_water = (200 < temp < 390) and (esi > 0.4 or (has_atm and 260 < temp < 380))
        else:
            has_water = (220 < temp < 370) and esi > 0.65
        snow_ln = 0.54 if has_water else (_snow_line(temp) if temp <= 200 else 0.28)
        seed = _planet_seed(planet)

        if use_explicit:
            atm_strength = float(np.clip(pressure / 3.0, 0, 1))
            gr, gg, gb   = _glow_rgb(oxygen)
            cloud_op  = float(np.clip((pressure - 0.05) * 0.80, 0, 0.92))
            cloud_cov = float(np.clip(0.55 - pressure * 0.04, 0.20, 0.75))
        else:
            atm_strength = 0.35
            if temp > 500:
                atm_strength, gr, gg, gb = 0.20, 0.85, 0.45, 0.15
            elif temp > 350:
                atm_strength, gr, gg, gb = 0.25, 0.70, 0.55, 0.25
            elif esi > 0.4:
                atm_strength, gr, gg, gb = 0.35, 0.35, 0.63, 1.0
            else:
                atm_strength, gr, gg, gb = 0.15, 0.60, 0.60, 0.65
            cloud_op     = 0.0
            cloud_cov    = 0.95

        planet_block = f"""
const planetMat = new THREE.ShaderMaterial({{
    uniforms: {{
        uLight:     {{ value: new THREE.Color('{light_hex}') }},
        uOcean:     {{ value: new THREE.Color('{dark_hex}') }},
        uSun:       {{ value: sunDir.clone() }},
        uRoughness: {{ value: {roughness:.3f} }},
        uSnowLine:  {{ value: {snow_ln:.3f} }},
        uSeed:      {{ value: {seed:.4f} }},
        uWater:     {{ value: {1.0 if has_water else 0.0:.1f} }},
    }},
    vertexShader: surfVert, fragmentShader: surfFrag,
}});
scene.add(new THREE.Mesh(new THREE.SphereGeometry(1, 64, 64), planetMat));
"""
        cloud_block = ""
        if cloud_op > 0.02:
            cloud_block = f"""
const cloudMat = new THREE.ShaderMaterial({{
    uniforms: {{
        uSun:      {{ value: sunDir.clone() }},
        uCoverage: {{ value: {cloud_cov:.3f} }},
        uOpacity:  {{ value: {cloud_op:.3f} }},
    }},
    vertexShader: cloudVert, fragmentShader: cloudFrag,
    transparent: true, depthWrite: false,
}});
scene.add(new THREE.Mesh(new THREE.SphereGeometry(1.04, 64, 64), cloudMat));
"""

    atm_block = ""
    if atm_strength > 0:
        atm_block = f"""
const atmMat = new THREE.ShaderMaterial({{
    uniforms: {{
        uGlow:     {{ value: new THREE.Color({gr:.4f}, {gg:.4f}, {gb:.4f}) }},
        uStrength: {{ value: {atm_strength:.3f} }},
    }},
    vertexShader: atmVert, fragmentShader: atmFrag,
    side: THREE.BackSide, blending: THREE.AdditiveBlending,
    transparent: true, depthWrite: false,
}});
scene.add(new THREE.Mesh(new THREE.SphereGeometry(1.07, 64, 64), atmMat));
"""

    return f"""<!DOCTYPE html>
<html>
<head>
<style>*{{margin:0;padding:0;box-sizing:border-box;}}body{{background:#0a0a1a;overflow:hidden;}}canvas{{display:block;width:100%;}}</style>
</head>
<body>
<canvas id="c"></canvas>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const canvas = document.getElementById('c');
const W = window.innerWidth || document.documentElement.clientWidth || 600;
const H = {height};
const renderer = new THREE.WebGLRenderer({{canvas, alpha: true, antialias: true}});
renderer.setSize(W, H);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42, W / H, 0.1, 100);
camera.position.set(0, 0, 3.0);
const controls = new THREE.OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.06;
controls.autoRotate = true;    controls.autoRotateSpeed = 0.6;
controls.minDistance = 1.8;    controls.maxDistance = 6.0;
const sun = new THREE.DirectionalLight(0xffffff, 1.3);
sun.position.set(5, 2, 4); scene.add(sun);
scene.add(new THREE.AmbientLight(0x112244, 0.35));
const sunDir = new THREE.Vector3(5, 2, 4).normalize();
const surfVert  = `{_VERT_SURF}`;
const surfFrag  = `{_FRAG_SURF}`;
const cloudVert = `{_VERT_CLOUD}`;
const cloudFrag = `{_FRAG_CLOUD}`;
const atmVert   = `{_VERT_ATM}`;
const atmFrag   = `{_FRAG_ATM}`;
const gasVert   = `{_VERT_GAS}`;
const gasFrag   = `{_FRAG_GAS}`;
{planet_block}
{cloud_block}
{atm_block}
function animate() {{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}}
animate();
window.addEventListener('resize', () => {{
    const nW = window.innerWidth || document.documentElement.clientWidth || 600;
    camera.aspect = nW / H;
    camera.updateProjectionMatrix();
    renderer.setSize(nW, H);
}});
</script>
</body>
</html>"""