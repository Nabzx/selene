# Demo script

Three minutes, live, real inputs. Rehearse at least twice.

## Before going on

- Backend running, view open, sound on low.
- `data/episodes.json` from the morning training run loaded.
- Cached safety spec loaded and labelled cached.
- Backup screen recording open in another tab.

## Script

| Time | On screen | Say |
|---|---|---|
| 0:00 | Far side at night, relay orbiter crossing, no Earth | "This is the far side of the Moon at night. Earth is never in view, it's minus 170, and nothing here can be flown by hand." |
| 0:20 | Safety spec panel | "Argon read the mission documents and wrote these rules. Each one is code we run." |
| 0:35 | Training replay, episode 1 crashing, episode 500 hopping cleanly | "The hopper learns the terrain from scratch. When it plateaued, Argon rewrote its reward." |
| 1:05 | World-model ghost drifting from the real hopper on a rim | "This ghost is where the hopper's own model thinks it will land. Here it's wrong. Argon explains why." |
| 1:25 | Argon plans a new route, trajectory cloud, red cluster | "Live: Argon plans tonight's route. We fly it a thousand times. Some runs crash here." |
| 1:50 | Revision, cloud turns green | "Argon reads the failures and moves the hop. Every run passes." |
| 2:15 | Relay window opens, approve, flight | "Only in a relay window, and only with a person's approval, does it go." |
| 2:45 | Pipeline diagram | "Learn the world, plan inside it, prove it, then act." |

## If Argon is slow

Start the route call in background mode at 0:35 while the training replay plays. Fall back to the cached route, labelled cached, at 1:40.
