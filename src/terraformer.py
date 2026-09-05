import json
import anthropic
from dataclasses import dataclass
from typing import List

from .features import (
      EARTH_EQT_K, EARTH_RADIUS_RE, EARTH_MASS_ME,
      EARTH_DENSITY_GCM3, EARTH_INSOL
  )

SYSTEM_PROMPT = """You are an expert astrobiologist and terraforming engineer with deep knowledge of planetary science, atmospheric chemistry, and the game Terragenesis. You
  generate scientifically grounded terraforming roadmaps that mirror Terragenesis game mechanics wherever applicable.

  Always respond with valid JSON matching the exact schema requested. Be specific about technologies, durations, and challenges. Reference real terraforming concepts (solar
  mirrors, greenhouse gas injection, magnetic field generators, lithopanspermia) and map them to Terragenesis mechanics (Pressure, Temperature, Water, Oxygen, Biomass sliders)."""
  
  
@dataclass
class TerraformingPhase:
    phase: int
    name: str
    duration: str
    objective: str
    technologies: List[str]
    terragenesis_mechanic: str


@dataclass
class TerraformingPlan:
    planet_name: str
    esi_score: float
    difficulty: str
    timeline_years: str
    key_challenges: List[str]
    phases: List[TerraformingPhase]
    
def generate_terraforming_plan(planet: dict, client: anthropic.Anthropic)->TerraformingPlan:
    eqt = float(planet.get("pl_eqt") or EARTH_EQT_K)
    radius = float(planet.get("pl_rade") or EARTH_RADIUS_RE)
    mass = float(planet.get("pl_bmasse") or EARTH_MASS_ME)
    insol = float(planet.get("insol_flux") or EARTH_INSOL)
    esi = float(planet.get("esi") or 0.0)
    in_hz = bool(planet.get("in_hz", 0))
    
    user_prompt = f"""Analyze this exoplanet and generate a Terragenesis-style terraforming roadmap.

  PLANET: {planet.get('pl_name', 'Unknown')}
  STAR: {planet.get('hostname', 'Unknown')} (Teff: {planet.get('st_teff', '?')}K, Type: {planet.get('st_spectype', '?')})
  ESI Score: {esi:.3f} (Earth = 1.000)

  CURRENT PARAMETERS vs EARTH:
  - Equilibrium Temperature: {eqt:.0f}K (Earth: {EARTH_EQT_K}K, delta: {eqt - EARTH_EQT_K:+.0f}K)
  - Planet Radius: {radius:.2f} Earth radii
  - Planet Mass: {mass:.2f} Earth masses
  - Insolation Flux: {insol:.3f} (Earth = 1.0)
  - In Habitable Zone: {in_hz}
  - Orbital Eccentricity: {planet.get('pl_orbeccen', 'unknown')}
  - Stellar Age: {planet.get('st_age', 'unknown')} Gyr

  Generate a JSON terraforming plan with this EXACT structure (no markdown, just JSON):
  {{
    "difficulty": "Easy|Moderate|Hard|Extreme|Impossible",
    "timeline_years": "e.g. 800-2500 years",
    "key_challenges": ["challenge 1", "challenge 2", "challenge 3"],
    "phases": [
      {{
        "phase": 1,
        "name": "Phase name",
        "duration": "X-Y years",
        "objective": "One sentence describing what this phase achieves",
        "technologies": ["tech 1", "tech 2", "tech 3"],
        "terragenesis_mechanic": "Which Terragenesis slider or mechanic this maps to"
      }}
    ]
  }}

  Maximum 4 phases. Be scientifically specific."""
    response=client.messages.create(
        model='claude-sonnet-4-6',
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "context": user_prompt}]
    )
    
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    
    data = json.loads(text)
    phases = [TerraformingPhase(**p) for p in data["phases"]]

    return TerraformingPlan(
        planet_name=planet.get("pl_name", "Unknown"),
        esi_score=esi,
        difficulty=data["difficulty"],
        timeline_years=data["timeline_years"],
        key_challenges=data["key_challenges"],
        phases=phases,
    )