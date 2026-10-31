# Selene — 3-Minute Pitch Script

**Total Time:** 3 minutes (~370 words + Demo Placeholder)

---

## 1. The Hook & Problem (0:00 – 0:45)

"A long time ago in a galaxy far, far away... space was full of autonomous droids hopping across alien worlds.

Today, that's no longer science fiction. Right now, humanity is moving to space at breakneck speed. **Space mining, orbital data centres, lunar bases, space defence**—space is the next great frontier.

And our first foothold is the Moon—specifically deep, shadowed craters like **Daedalus Crater** on the lunar far side, which hold billions of tonnes of water ice and rare minerals.

But there's a huge problem:
1. **Wheeled rovers get stuck** or flip over on steep $30^\circ$ crater walls.
2. **There is zero direct radio contact with Earth** on the far side, and temperatures plunge to $-180^\circ\text{C}$.

Rocket-powered **jumping drones—lunar hoppers—can leap straight over these cliffs**. But you can't drive them with a joystick from Earth, and you can't let a raw AI hallucinate a flight plan—because one bad coordinate in lunar gravity destroys a \$50 million mission.

That's why we built **Selene**: the autonomous mission brain and physics safety gate for fleets of lunar hopping drones."

---

## 2. Market Opportunity (0:45 – 1:10)

"Selene sits at the convergence of three explosive markets:
- **The Global Space Economy:** \$630 billion today, projected by Morgan Stanley and McKinsey to hit **\$1.8 trillion by 2035**, including a **\$170 billion lunar economy**.
- **Autonomous Drone Swarms:** Growing from **\$40 billion to \$115 billion by 2030**.
- **Physical AI & World Models:** A **\$150+ billion market** bringing AI out of chat boxes and into real-world hardware."

---

## 3. What We Built & Tech Stack (1:10 – 1:55)

"To solve this, we combined two systems that work together in a closed loop: **an AI Mission Orchestrator** and **a Physics & Reinforcement Learning Engine**.

Our tech stack brings together **Gemini 4 Argon** via the Interactions API, **PyTorch**, **PettingZoo** and **Gymnasium** for multi-agent environments, **NumPy**, **Pydantic**, and **real NASA LOLA satellite datasets**.

1. **The AI Orchestrator (Gemini 4 Argon):** Reads raw NASA mission rules, extracts hard safety constraints, and plans coordinated flight paths for 4 hoppers so they explore craters without colliding or jamming the single satellite relay channel.
2. **The Physics Engine, MARL & World Models:** Built on real NASA Daedalus Crater topography, our engine simulates lunar gravity ($1.62\,\text{m/s}^2$), **ballistic trajectories**, the **Tsiolkovsky rocket equation** for fuel burn, and **thermal battery drain**.
   Using **PyTorch**, we trained **Multi-Agent Reinforcement Learning (MAPPO)** alongside a **5-network Ensemble World Model** so the swarm learns to dodge hazards and maximise science scans—boosting fleet reward from **$-2.5$ to $+25.6$**.
3. **The 1,000-Rollout Monte Carlo Safety Gate:** Before any AI plan is allowed to uplink, our simulator stress-tests it **1,000 times in under 2 seconds** against random thruster misfires and sensor noise. If it fails even once, Argon automatically rewrites the plan until it hits a **100% safety pass rate**, before halting for human sign-off."

---

## 4. Live Demo (1:55 – 2:30)

**[ DEMO PLACEHOLDER ]**

---

## 5. Future Use Cases & Next Steps (2:30 – 3:00)

"Beyond the Moon and asteroids, this exact 'AI Planner + Physics Verifier' stack works anywhere on Earth where GPS and radio fail: **deep-sea infrastructure, underground mining, and disaster search-and-rescue swarms**.

With more time, our next upgrades are:
1. **Full 6-Degree-of-Freedom (6-DOF) landing physics** to model individual landing legs sinking into lunar dust.
2. **Direct Neural Feedback**, letting Argon query our PyTorch Reinforcement Learning critic directly while brainstorming flight paths.

Thank you!"
